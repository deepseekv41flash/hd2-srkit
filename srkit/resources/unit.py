"""``mesh``/``unit`` resource codec - the Stingray model format.

Cross checked against ``stingray/unit.py`` (2600 lines in the add-on).  The toc
blob is a header of eight 32/64 bit sections offsets followed by the sections::

    0x00 u64 unknown ref 1            0x08 u64 bones resource id
    0x10 u64 geometry group id        0x18 u64 unknown ref 2
    0x20 u64 state machine id         0x28 u32 unknown
    0x2C u32 version                  0x30 u32 lod group list offset
    0x34 u32 transform info offset    0x38 u32 light list offset
    0x3C u32 pre light list offset    0x40 u32 wwise callback offset
    0x44  8 unknown bytes             0x4C u32 customization info offset
    0x50 u32 unknown header 1 offset  0x54 u32 connecting bones offset
    0x58 u32 bone info offset         0x5C u32 stream info offset
    0x60 u32 ending offset            0x64 u32 mesh info offset
    0x68 u64 unknown                  0x70 u32 materials offset
    0x74 12 padding bytes             -> 0x80

Sections are written in the order their offsets point at.  Whatever is not
understood (LOD group list, customization blob, bone info matrices, ...) is kept
as an opaque range and written back verbatim, so a resource that is not edited
round trips byte for byte.

The vertex/index buffers live in the *gpu* blob and are sliced out of it using
the stream info tables; ``--decode-vertices`` additionally writes human readable
per mesh vertex/index dumps (views, never read back).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from ..container.package import TocEntry
from ..errors import CodecError, FormatError
from ..util.binary import Stream, align_up
from ..util.ids import NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["UnitCodec", "codec", "component_size", "component_format_name"]

HEADER_SIZE = 0x80
OLD_UNIT_VERSION = 10800437
NEW_UNIT_VERSION = 10800438

# --------------------------------------------------------------------------- #
# vertex component tables (mirrors StreamComponentFormat / StreamComponentType)
# --------------------------------------------------------------------------- #
COMPONENT_TYPES = {
    0: "position",
    1: "normal",
    2: "tangent",
    3: "bitangent",
    4: "uv",
    5: "color",
    6: "bone_index",
    7: "bone_weight",
}

#: format id -> (name, byte size) for current units
COMPONENT_FORMATS = {
    0: ("float", 4),
    1: ("vec2_float", 8),
    2: ("vec3_float", 12),
    3: ("vec4_float", 16),
    4: ("rgba_r8g8b8a8", 4),
    21: ("uint32", 4),
    22: ("vec2_uint32", 8),
    23: ("vec3_uint32", 12),
    24: ("vec4_uint32", 16),
    25: ("uint8", 1),
    26: ("vec2_uint8", 2),
    27: ("vec3_uint8", 3),
    28: ("vec4_uint8", 4),
    29: ("vec4_1010102", 4),
    30: ("unk_normal", 4),
    32: ("float16", 2),
    33: ("vec2_half", 4),
    34: ("vec3_half", 6),
    35: ("vec4_half", 8),
}

#: format id -> (name, byte size) for units of version 10800437 and older
COMPONENT_FORMATS_OLD = {
    0: ("float", 4),
    1: ("vec2_float", 8),
    2: ("vec3_float", 12),
    3: ("vec4_float", 16),
    4: ("rgba_r8g8b8a8", 4),
    17: ("uint32", 4),
    18: ("vec2_uint32", 8),
    19: ("vec3_uint32", 12),
    20: ("vec4_uint32", 16),
    21: ("uint8", 1),
    22: ("vec2_uint8", 2),
    23: ("vec3_uint8", 3),
    24: ("vec4_uint8", 4),
    25: ("vec4_1010102", 4),
    26: ("unk_normal", 4),
    28: ("float16", 2),
    29: ("vec2_half", 4),
    30: ("vec3_half", 6),
    31: ("vec4_half", 8),
}


def _format_table(version: int) -> dict[int, tuple[str, int]]:
    return COMPONENT_FORMATS_OLD if version == OLD_UNIT_VERSION else COMPONENT_FORMATS


def component_format_name(format_id: int, version: int = NEW_UNIT_VERSION) -> str:
    entry = _format_table(version).get(int(format_id))
    return entry[0] if entry else f"unknown_format_{int(format_id)}"


def component_size(format_id: int, version: int = NEW_UNIT_VERSION) -> int:
    entry = _format_table(version).get(int(format_id))
    if entry is None:
        raise CodecError(
            f"unit: unknown vertex component format {format_id} (version {version})"
        )
    return entry[1]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _hex(data: bytes) -> str:
    return bytes(data).hex()


def _octahedral_decode(x: float, y: float) -> tuple[float, float, float]:
    z = 1.0 - abs(x) - abs(y)
    if z < 0.0:
        sign_x = 1.0 if x >= 0 else -1.0
        sign_y = 1.0 if y >= 0 else -1.0
        x, y = (1.0 - abs(y)) * sign_x, (1.0 - abs(x)) * sign_y
    length = (x * x + y * y + z * z) ** 0.5 or 1.0
    return x / length, y / length, z / length


def decode_packed_oct_normal(value: int) -> tuple[float, float, float]:
    r10 = value & 0x3FF
    g10 = (value >> 10) & 0x3FF
    return _octahedral_decode(r10 * (2.0 / 1023.0) - 1.0, g10 * (2.0 / 1023.0) - 1.0)


def _ten_bit_unsigned(value: int) -> list[float]:
    x = value & 1023
    y = (value >> 10) & 1023
    z = (value >> 20) & 1023
    w = (value >> 30) & 3
    return [x / 1023.0, y / 1023.0, z / 1023.0, w / 3.0]


# --------------------------------------------------------------------------- #
# structures
# --------------------------------------------------------------------------- #
@dataclass
class UnitHeader:
    unk_ref1: int = 0
    bones_ref: int = 0
    geometry_group_ref: int = 0
    unk_ref2: int = 0
    state_machine_ref: int = 0
    unknown_28: int = 0
    version: int = NEW_UNIT_VERSION
    lod_group_list_offset: int = 0
    transform_info_offset: int = 0
    light_list_offset: int = 0
    pre_light_list_offset: int = 0
    wwise_callback_offset: int = 0
    header_data2: bytes = b"\x00" * 8
    customization_offset: int = 0
    unk_header1_offset: int = 0
    connecting_bones_offset: int = 0
    bone_info_offset: int = 0
    stream_info_offset: int = 0
    ending_offset: int = 0
    mesh_info_offset: int = 0
    unknown_68: int = 0
    materials_offset: int = 0
    padding: bytes = b"\x00" * 12

    @classmethod
    def parse(cls, data: bytes) -> "UnitHeader":
        if len(data) < HEADER_SIZE:
            raise FormatError(f"unit: toc blob too small ({len(data)} bytes)")
        (
            unk_ref1, bones_ref, geometry_group_ref, unk_ref2, state_machine_ref,
        ) = struct.unpack_from("<QQQQQ", data, 0x00)
        (
            unknown_28, version, lod_group_list_offset, transform_info_offset,
            light_list_offset, pre_light_list_offset, wwise_callback_offset,
        ) = struct.unpack_from("<IIIIIII", data, 0x28)
        header_data2 = data[0x44:0x4C]
        (
            customization_offset, unk_header1_offset, connecting_bones_offset,
            bone_info_offset, stream_info_offset, ending_offset, mesh_info_offset,
        ) = struct.unpack_from("<IIIIIII", data, 0x4C)
        unknown_68 = struct.unpack_from("<Q", data, 0x68)[0]
        materials_offset = struct.unpack_from("<I", data, 0x70)[0]
        return cls(
            unk_ref1=unk_ref1,
            bones_ref=bones_ref,
            geometry_group_ref=geometry_group_ref,
            unk_ref2=unk_ref2,
            state_machine_ref=state_machine_ref,
            unknown_28=unknown_28,
            version=version,
            lod_group_list_offset=lod_group_list_offset,
            transform_info_offset=transform_info_offset,
            light_list_offset=light_list_offset,
            pre_light_list_offset=pre_light_list_offset,
            wwise_callback_offset=wwise_callback_offset,
            header_data2=header_data2,
            customization_offset=customization_offset,
            unk_header1_offset=unk_header1_offset,
            connecting_bones_offset=connecting_bones_offset,
            bone_info_offset=bone_info_offset,
            stream_info_offset=stream_info_offset,
            ending_offset=ending_offset,
            mesh_info_offset=mesh_info_offset,
            unknown_68=unknown_68,
            materials_offset=materials_offset,
            padding=data[0x74:0x80],
        )

    def to_bytes(self) -> bytes:
        out = bytearray(HEADER_SIZE)
        struct.pack_into(
            "<QQQQQ", out, 0x00,
            self.unk_ref1, self.bones_ref, self.geometry_group_ref,
            self.unk_ref2, self.state_machine_ref,
        )
        struct.pack_into(
            "<IIIIIII", out, 0x28,
            self.unknown_28, self.version, self.lod_group_list_offset,
            self.transform_info_offset, self.light_list_offset,
            self.pre_light_list_offset, self.wwise_callback_offset,
        )
        out[0x44:0x4C] = self.header_data2[:8].ljust(8, b"\x00")
        struct.pack_into(
            "<IIIIIII", out, 0x4C,
            self.customization_offset, self.unk_header1_offset,
            self.connecting_bones_offset, self.bone_info_offset,
            self.stream_info_offset, self.ending_offset, self.mesh_info_offset,
        )
        struct.pack_into("<Q", out, 0x68, self.unknown_68)
        struct.pack_into("<I", out, 0x70, self.materials_offset)
        out[0x74:0x80] = self.padding[:12].ljust(12, b"\x00")
        return bytes(out)


@dataclass
class StreamInfo:
    component_info_id: int = 0
    components: list[dict] = field(default_factory=list)
    component_region_tail: bytes = b""
    num_components_declared: int = 0
    vertex_buffer_id: int = 0
    vertex_buffer_unk1: int = 0
    num_vertices: int = 0
    vertex_stride: int = 0
    vertex_buffer_unk2: int = 0
    vertex_buffer_unk3: int = 0
    index_buffer_id: int = 0
    index_buffer_unk1: int = 0
    num_indices: int = 0
    index_buffer_type: int = 0
    index_buffer_unk2: int = 0
    index_buffer_unk3: int = 0
    vertex_buffer_offset: int = 0
    vertex_buffer_size: int = 0
    index_buffer_offset: int = 0
    index_buffer_size: int = 0
    unk_ending_bytes: bytes = b"\x00" * 16
    offset: int = 0            # relative to the stream info section start
    section_offset: int = 0    # original value of ``offset`` (layout hint)
    unknown_offsets: list[int] = field(default_factory=list)

    COMPONENT_REGION = 320


# --------------------------------------------------------------------------- #
# light list
# --------------------------------------------------------------------------- #
@dataclass
class Light:
    name_hash: int = 0
    bone_index: int = 0
    color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    intensity: float = 1.0
    falloff_start: float = 0.0
    falloff_end: float = 0.0
    falloff_exp: float = 1.0
    start_angle: float = 0.0
    end_angle: float = 0.0
    unknown0: float = 0.0
    shadow_bias: float = 0.4
    unknown1: list[float] = field(default_factory=lambda: [0.0] * 5)
    flags: int = 0
    light_type: int = 0
    unknown2: bytes = b"\x00" * 32

    LIGHT_TYPES = {0: "omni", 1: "spot", 2: "box", 3: "directional"}

    @classmethod
    def parse(cls, stream: Stream) -> "Light":
        light = cls()
        light.name_hash = stream.uint32()
        light.bone_index = stream.uint32()
        light.color = [stream.float32() for _ in range(3)]
        light.intensity = stream.float32()
        light.falloff_start = stream.float32()
        light.falloff_end = stream.float32()
        light.falloff_exp = stream.float32()
        light.start_angle = stream.float32()
        light.end_angle = stream.float32()
        light.unknown0 = stream.float32()
        light.shadow_bias = stream.float32()
        light.unknown1 = [stream.float32() for _ in range(5)]
        light.flags = stream.uint8()
        stream.read(3)
        light.light_type = stream.uint32()
        light.unknown2 = bytes(stream.bytes(b"", 32))
        return light

    def to_bytes(self) -> bytes:
        stream = Stream.writer()
        stream.uint32(self.name_hash)
        stream.uint32(self.bone_index)
        for value in self.color:
            stream.float32(value)
        stream.float32(self.intensity)
        stream.float32(self.falloff_start)
        stream.float32(self.falloff_end)
        stream.float32(self.falloff_exp)
        stream.float32(self.start_angle)
        stream.float32(self.end_angle)
        stream.float32(self.unknown0)
        stream.float32(self.shadow_bias)
        for value in (self.unknown1 + [0.0] * 5)[:5]:
            stream.float32(value)
        stream.uint8(self.flags)
        stream.write(b"\x00\x00\x00")
        stream.uint32(self.light_type)
        stream.write(self.unknown2[:32].ljust(32, b"\x00"))
        return stream.getvalue()

    def to_json(self, names: NameTable) -> dict:
        return {
            "name_hash": str(self.name_hash),
            "name": names.friendly(self.name_hash) or "",
            "bone_index": self.bone_index,
            "type": self.LIGHT_TYPES.get(self.light_type, str(self.light_type)),
            "color": self.color,
            "intensity": self.intensity,
            "falloff_start": self.falloff_start,
            "falloff_end": self.falloff_end,
            "falloff_exp": self.falloff_exp,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "shadow_bias": self.shadow_bias,
            "flags": self.flags,
            "unknown0": self.unknown0,
            "unknown1": self.unknown1,
            "unknown2_hex": _hex(self.unknown2),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Light":
        light = cls()
        light.name_hash = int(data.get("name_hash", 0))
        light.bone_index = int(data.get("bone_index", 0))
        light.color = [float(v) for v in (data.get("color") or [0, 0, 0])][:3]
        light.intensity = float(data.get("intensity", 1.0))
        light.falloff_start = float(data.get("falloff_start", 0.0))
        light.falloff_end = float(data.get("falloff_end", 0.0))
        light.falloff_exp = float(data.get("falloff_exp", 1.0))
        light.start_angle = float(data.get("start_angle", 0.0))
        light.end_angle = float(data.get("end_angle", 0.0))
        light.unknown0 = float(data.get("unknown0", 0.0))
        light.shadow_bias = float(data.get("shadow_bias", 0.4))
        light.unknown1 = [float(v) for v in (data.get("unknown1") or [0] * 5)][:5]
        light.flags = int(data.get("flags", 0))
        reverse = {value: key for key, value in cls.LIGHT_TYPES.items()}
        kind = data.get("type", 0)
        light.light_type = reverse.get(kind, int(kind) if str(kind).isdigit() else 0)
        hexed = data.get("unknown2_hex", "") or ""
        light.unknown2 = bytes.fromhex(hexed).ljust(32, b"\x00")[:32]
        return light


class LightList:
    def __init__(self):
        self.unknown0: list[int] = [0, 0, 0]
        self.lights: list[Light] = []

    @classmethod
    def parse(cls, stream: Stream) -> "LightList":
        result = cls()
        count = stream.uint32()
        result.unknown0 = [stream.uint32() for _ in range(3)]
        result.lights = [Light.parse(stream) for _ in range(count)]
        return result

    def to_bytes(self) -> bytes:
        stream = Stream.writer()
        stream.uint32(len(self.lights))
        for value in (self.unknown0 + [0, 0, 0])[:3]:
            stream.uint32(value)
        for light in self.lights:
            stream.write(light.to_bytes())
        return stream.getvalue()


# --------------------------------------------------------------------------- #
# the codec
# --------------------------------------------------------------------------- #
class UnitCodec(ResourceCodec):
    name = "unit"
    type_ids: tuple[int, ...] = (0xE0A48D0BE9A7453F,)  # TYPE_UNIT
    handles = ("toc", "gpu", "stream")

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def decode(
        self,
        entry: TocEntry,
        names: NameTable,
        options: dict[str, Any] | None = None,
    ) -> DecodedResource:
        options = options or {}
        data = bytes(entry.toc_data)
        header = UnitHeader.parse(data)
        version = header.version

        ir = self.base_ir(entry, "unit")
        ir["version"] = version
        ir["version_supported"] = version in (OLD_UNIT_VERSION, NEW_UNIT_VERSION)
        ir["read_only"] = False
        ir["header"] = {
            "unk_ref1": str(header.unk_ref1),
            "bones_ref": str(header.bones_ref),
            "geometry_group_ref": str(header.geometry_group_ref),
            "unk_ref2": str(header.unk_ref2),
            "state_machine_ref": str(header.state_machine_ref),
            "unknown_28": header.unknown_28,
            "header_data2_hex": _hex(header.header_data2),
            "unknown_68": str(header.unknown_68),
            "padding_hex": _hex(header.padding),
        }

        offsets = {
            "wwise_callback": header.wwise_callback_offset,
            "pre_light_list": header.pre_light_list_offset,
            "light_list": header.light_list_offset,
            "lod_group_list": header.lod_group_list_offset,
            "transform_info": header.transform_info_offset,
            "customization": header.customization_offset,
            "unk_header1": header.unk_header1_offset,
            "connecting_bones": header.connecting_bones_offset,
            "bone_info": header.bone_info_offset,
            "stream_info": header.stream_info_offset,
            "mesh_info": header.mesh_info_offset,
            "materials": header.materials_offset,
            "ending": header.ending_offset,
        }
        ir["section_offsets"] = {key: value for key, value in offsets.items() if value}

        # ---- opaque ranges ------------------------------------------- #
        # Sections have no explicit length: a section ends where the next
        # section in file order begins (the rule the add-on uses, generalised so
        # it also works for units that are missing optional sections).
        known_offsets = sorted(
            value for value in (
                header.wwise_callback_offset,
                header.pre_light_list_offset,
                header.light_list_offset,
                header.lod_group_list_offset,
                header.transform_info_offset,
                header.customization_offset,
                header.unk_header1_offset,
                header.connecting_bones_offset,
                header.bone_info_offset,
                header.stream_info_offset,
                header.mesh_info_offset,
                header.materials_offset,
                header.ending_offset,
            ) if value
        )

        def section_end(start: int) -> int:
            for value in known_offsets:
                if value > start:
                    return value
            return len(data)

        opaque: dict[str, bytes] = {}
        for name, start in (
            ("wwise_callback", header.wwise_callback_offset),
            ("pre_light_list", header.pre_light_list_offset),
            ("transform_info", header.transform_info_offset),
            ("customization", header.customization_offset),
            ("unk_header1", header.unk_header1_offset),
            ("connecting_bones", header.connecting_bones_offset),
            ("bone_info", header.bone_info_offset),
        ):
            if not start:
                continue
            end = section_end(start)
            if end < start:
                raise FormatError(f"unit: {name} range is inverted")
            opaque[name] = data[start:end]

        # ---- light list ---------------------------------------------- #
        light_list = LightList()
        if header.light_list_offset:
            stream = Stream.reader(data)
            stream.seek(header.light_list_offset)
            light_list = LightList.parse(stream)
            ir["lights"] = [light.to_json(names) for light in light_list.lights]
            ir["lights_unknown0"] = light_list.unknown0
            lights_end = stream.tell()
        else:
            lights_end = HEADER_SIZE

        # ---- lod group list (implicit length) ------------------------ #
        lod_end = section_end(lights_end)
        if lod_end > lights_end:
            opaque["lod_group_list"] = data[lights_end:lod_end]
        ir["lod_group_list_offset_hint"] = header.lod_group_list_offset

        # ---- stream info --------------------------------------------- #
        streams: list[StreamInfo] = []
        if header.stream_info_offset and header.mesh_info_offset:
            streams = self._parse_stream_info(
                data, header.stream_info_offset, header.mesh_info_offset, version
            )
        ir["streams"] = [self._stream_to_json(item, version, names) for item in streams]

        # ---- mesh info (2016 -> materials/ending) --------------------- #
        meshes: list[dict] = []
        materials: list[dict] = []
        mesh_unknowns: list[int] = []
        struct_end = header.mesh_info_offset
        if header.mesh_info_offset:
            meshes, struct_end, mesh_unknowns = self._parse_mesh_table(
                data, header, names
            )
        if header.materials_offset:
            materials, struct_end = self._parse_material_table(data, header, names)

        ending_offset = header.ending_offset
        has_ending_marker = bool(ending_offset)
        # bytes not accounted for by any section: everything between the end of
        # the last parsed section and the end of the file / ending marker
        opaque_end = 0
        for start in (
            header.wwise_callback_offset, header.pre_light_list_offset,
            header.transform_info_offset, header.customization_offset,
            header.unk_header1_offset, header.connecting_bones_offset,
            header.bone_info_offset,
        ):
            if start:
                opaque_end = max(opaque_end, section_end(start))
        tail_start = max(struct_end, opaque_end)
        unreversed2 = b""
        ending_bytes = 0
        trailing = b""
        if has_ending_marker:
            if ending_offset >= tail_start:
                unreversed2 = data[tail_start:ending_offset]
            if ending_offset + 8 <= len(data):
                ending_bytes = struct.unpack_from("<Q", data, ending_offset)[0]
                trailing = data[ending_offset + 8:]
        elif tail_start < len(data):
            unreversed2 = data[tail_start:]

        ir["meshes"] = meshes
        ir["materials"] = materials
        ir["mesh_unknown_ids"] = list(mesh_unknowns)
        ir["ending_bytes"] = ending_bytes
        ir["has_ending_marker"] = has_ending_marker
        ir["has_materials_section"] = bool(header.materials_offset)

        if header.transform_info_offset:
            ir["transforms"] = _parse_transform_view(
                data, header.transform_info_offset, names
            )

        # ---- decoded views / light structural data -------------------- #
        payloads: list[Payload] = []
        if entry.gpu_data:
            payloads.append(
                self.master_payload("gpu.bin", bytes(entry.gpu_data),
                                    note="vertex/index buffers (sliced by the stream info tables)")
            )
        if entry.stream_data:
            payloads.append(
                self.master_payload("stream.bin", bytes(entry.stream_data),
                                    note="streamed mesh data")
            )
        for name, blob in opaque.items():
            payloads.append(
                Payload(
                    f"opaque_{name}.bin",
                    blob,
                    editable=False,
                    note=f"opaque section '{name}' preserved verbatim",
                )
            )
        if unreversed2:
            payloads.append(
                self.master_payload("unreversed2.bin", unreversed2,
                                    note="bytes between the material table and the ending offset")
            )
        if trailing:
            payloads.append(
                self.master_payload("toc_tail.bin", trailing,
                                    note="bytes after the ending marker")
            )

        if header.customization_offset:
            ir["customization"] = _parse_customization(data[header.customization_offset:])

        if options.get("decode_vertices"):
            payloads.extend(
                self._vertex_views(entry, streams, ir["meshes"], version, names)
            )

        ir["opaque_sections"] = {
            name: {"size": len(blob), "sha256_prefix": _hex(blob[:8])}
            for name, blob in opaque.items()
        }
        ir["gpu_size"] = len(entry.gpu_data)
        ir["stream_size"] = len(entry.stream_data)
        return DecodedResource(ir, payloads)

    # ------------------------------------------------------------------ #
    def _parse_stream_info(
        self, data: bytes, start: int, end: int, version: int
    ) -> list[StreamInfo]:
        table = _format_table(version)
        stream = Stream.reader(data)
        stream.seek(start)
        num_streams = stream.uint32()
        offsets = [stream.uint32() for _ in range(num_streams)]
        unknown_offsets = [stream.uint32() for _ in range(num_streams)]
        unknown2 = stream.uint32()
        _ = unknown2

        streams: list[StreamInfo] = []
        for index in range(num_streams):
            item = StreamInfo()
            item.offset = offsets[index]
            item.section_offset = offsets[index]
            item.unknown_offsets = list(unknown_offsets)
            stream.seek(start + offsets[index])
            block_start = stream.tell()
            item.component_info_id = stream.uint64()
            component_start = stream.tell()
            stream.seek(component_start + StreamInfo.COMPONENT_REGION)
            item.num_components_declared = stream.uint64()
            item.vertex_buffer_id = stream.uint64()
            item.vertex_buffer_unk1 = stream.uint64()
            item.num_vertices = stream.uint32()
            item.vertex_stride = stream.uint32()
            item.vertex_buffer_unk2 = stream.uint64()
            item.vertex_buffer_unk3 = stream.uint64()
            item.index_buffer_id = stream.uint64()
            item.index_buffer_unk1 = stream.uint64()
            item.num_indices = stream.uint32()
            item.index_buffer_type = stream.uint32()
            item.index_buffer_unk2 = stream.uint64()
            item.index_buffer_unk3 = stream.uint64()
            item.vertex_buffer_offset = stream.uint32()
            item.vertex_buffer_size = stream.uint32()
            item.index_buffer_offset = stream.uint32()
            item.index_buffer_size = stream.uint32()
            item.unk_ending_bytes = bytes(stream.bytes(b"", 16))

            stream.seek(component_start)
            components = []
            for _ in range(item.num_components_declared):
                type_id = stream.uint32()
                format_id = stream.uint32()
                component_index = stream.uint32()
                unknown = stream.uint64()
                components.append(
                    {
                        "type": type_id,
                        "type_name": COMPONENT_TYPES.get(type_id, f"unknown_{type_id}"),
                        "format": format_id,
                        "format_name": table.get(format_id, (f"unknown_{format_id}", 0))[0],
                        "size": table.get(format_id, ("", 0))[1],
                        "index": component_index,
                        "unknown": str(unknown),
                    }
                )
            item.components = components
            used = item.num_components_declared * 20
            tail = data[component_start + used:component_start + StreamInfo.COMPONENT_REGION]
            item.component_region_tail = tail
            streams.append(item)

            if block_start + 0x1C0 > end + 16:
                raise FormatError("unit: stream info block exceeds its section")
        return streams

    def _parse_mesh_table(
        self, data: bytes, header: UnitHeader, names: NameTable
    ) -> tuple[list[dict], int, list[int]]:
        """Parse the mesh info section; returns (meshes, end offset, unknown ids)."""
        stream = Stream.reader(data)
        stream.seek(header.mesh_info_offset)
        num_meshes = stream.uint32()
        mesh_offsets = [stream.uint32() for _ in range(num_meshes)]
        mesh_unknowns = [stream.uint32() for _ in range(num_meshes)]
        meshes: list[dict] = []
        end = stream.tell()
        for index in range(num_meshes):
            stream.seek(header.mesh_info_offset + mesh_offsets[index])
            base = stream.tell()
            unk1 = stream.uint64()
            unk2 = bytes(stream.bytes(b"", 32))
            mesh_id = stream.uint32()
            unk3 = stream.uint32()
            transform_index = stream.uint32()
            unk4 = stream.uint32()
            lod_index = stream.int32()
            stream_index = stream.uint32()
            unk6 = bytes(stream.bytes(b"", 40))
            num_materials = stream.uint32()
            material_offset = stream.uint32()
            unk8 = stream.uint64()
            num_sections = stream.uint32()
            sections_offset = stream.uint32()
            material_ids = [stream.uint32() for _ in range(num_materials)]
            end = max(end, stream.tell())
            stream.seek(base + sections_offset)
            sections = []
            for _ in range(num_sections):
                material_index = stream.uint32()
                vertex_offset = stream.uint32()
                num_vertices = stream.uint32()
                index_offset = stream.uint32()
                num_indices = stream.uint32()
                group_index = stream.uint32()
                sections.append(
                    {
                        "material_index": material_index,
                        "material_slot_id": (
                            material_ids[material_index]
                            if material_index < len(material_ids) else 0
                        ),
                        "vertex_offset": vertex_offset,
                        "num_vertices": num_vertices,
                        "index_offset": index_offset,
                        "num_indices": num_indices,
                        "group_index": group_index,
                    }
                )
            end = max(end, stream.tell())
            meshes.append(
                {
                    "index": index,
                    "mesh_id": mesh_id,
                    "name": names.friendly(mesh_id) or "",
                    "lod_index": lod_index,
                    "stream_index": stream_index,
                    "transform_index": transform_index,
                    "material_slot_ids": material_ids,
                    "material_offset": material_offset,
                    "section_offset": mesh_offsets[index],
                    "sections": sections,
                    "unknown": {
                        "unk1": str(unk1),
                        "unk2_hex": _hex(unk2),
                        "unk3": unk3,
                        "unk4": unk4,
                        "unk6_hex": _hex(unk6),
                        "unk8": str(unk8),
                    },
                }
            )
        return meshes, end, mesh_unknowns

    def _parse_material_table(
        self, data: bytes, header: UnitHeader, names: NameTable
    ) -> tuple[list[dict], int]:
        """Parse the unit level material table; returns (materials, end offset)."""
        stream = Stream.reader(data)
        stream.seek(header.materials_offset)
        num_materials = stream.uint32()
        section_ids = [stream.uint32() for _ in range(num_materials)]
        material_ids = [stream.uint64() for _ in range(num_materials)]
        materials = [
            {
                "slot_id": section_ids[index],
                "material_id": str(material_ids[index]),
                "material_name": names.label(material_ids[index]) if material_ids[index] else "",
            }
            for index in range(num_materials)
        ]
        return materials, stream.tell()

    def _stream_to_json(self, item: StreamInfo, version: int, names: NameTable) -> dict:
        return {
            "component_info_id": str(item.component_info_id),
            "num_vertices": item.num_vertices,
            "vertex_stride": item.vertex_stride,
            "vertex_buffer": {
                "id": str(item.vertex_buffer_id),
                "unk1": str(item.vertex_buffer_unk1),
                "unk2": str(item.vertex_buffer_unk2),
                "unk3": str(item.vertex_buffer_unk3),
                "offset": item.vertex_buffer_offset,
                "size": item.vertex_buffer_size,
            },
            "index_buffer": {
                "id": str(item.index_buffer_id),
                "unk1": str(item.index_buffer_unk1),
                "unk2": str(item.index_buffer_unk2),
                "unk3": str(item.index_buffer_unk3),
                "type": item.index_buffer_type,
                "index_size": 4 if item.index_buffer_type == 1 else 2,
                "num_indices": item.num_indices,
                "offset": item.index_buffer_offset,
                "size": item.index_buffer_size,
            },
            "components": item.components,
            "component_region_tail_hex": _hex(item.component_region_tail),
            "section_offset": item.section_offset,
            "unknown_offsets": item.unknown_offsets,
            "unk_ending_bytes_hex": _hex(item.unk_ending_bytes),
        }

    # ------------------------------------------------------------------ #
    def _vertex_views(
        self,
        entry: TocEntry,
        streams: list[StreamInfo],
        meshes: list[dict],
        version: int,
        names: NameTable,
    ) -> list[Payload]:
        """Decoded vertex/index dumps (views only, never read back)."""
        if not entry.gpu_data or not streams:
            return []
        gpu = bytes(entry.gpu_data)
        payloads: list[Payload] = []
        for mesh in meshes:
            stream_index = mesh.get("stream_index", 0)
            if stream_index >= len(streams):
                continue
            info = streams[stream_index]
            sections = mesh.get("sections") or []
            if not sections or not info.vertex_stride:
                continue
            first = sections[0]
            base = info.vertex_buffer_offset + first["vertex_offset"] * info.vertex_stride
            count = first["num_vertices"] or info.num_vertices
            vertices = []
            for vidx in range(count):
                offset = base + vidx * info.vertex_stride
                if offset + info.vertex_stride > len(gpu):
                    break
                vertices.append(
                    self._decode_vertex(gpu, offset, info.components, version)
                )
            index_stride = 4 if info.index_buffer_type == 1 else 2
            index_start = info.index_buffer_offset + first["index_offset"] * index_stride
            total_indices = sum(section["num_indices"] for section in sections)
            indices = []
            for iidx in range(total_indices):
                offset = index_start + iidx * index_stride
                if offset + index_stride > len(gpu):
                    break
                indices.append(
                    int.from_bytes(gpu[offset:offset + index_stride], "little")
                )
            payloads.append(
                Payload(
                    f"mesh{mesh['index']}_vertices.json",
                    _json_bytes(vertices),
                    view=True,
                    note=" decoded vertex list (view)",
                )
            )
            payloads.append(
                Payload(
                    f"mesh{mesh['index']}_indices.json",
                    _json_bytes(indices),
                    view=True,
                    note="decoded index list (view)",
                )
            )
        return payloads

    @staticmethod
    def _decode_vertex(
        gpu: bytes, offset: int, components: list[dict], version: int
    ) -> dict:
        result: dict[str, Any] = {}
        cursor = offset
        uv_index = 0
        bone_index_count = 0
        for component in components:
            size = component.get("size") or component_size(component["format"], version)
            raw = gpu[cursor:cursor + size]
            name = component["type_name"]
            if name == "uv":
                name = f"uv{component.get('index', uv_index)}"
                uv_index += 1
            elif name == "bone_index":
                name = f"bone_index{bone_index_count}"
                bone_index_count += 1
            result[name] = _decode_component(raw, component["format"], version)
            cursor += size
        return result

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        header = UnitHeader.parse(bytes(entry.toc_data)) if entry.toc_data else UnitHeader()
        version = int(ir.get("version", header.version))

        # keep the header bookkeeping that the IR does not model
        header_ir = ir.get("header") or {}
        header.unk_ref1 = int(header_ir.get("unk_ref1", header.unk_ref1))
        header.bones_ref = int(header_ir.get("bones_ref", header.bones_ref))
        header.geometry_group_ref = int(
            header_ir.get("geometry_group_ref", header.geometry_group_ref)
        )
        header.unk_ref2 = int(header_ir.get("unk_ref2", header.unk_ref2))
        header.state_machine_ref = int(
            header_ir.get("state_machine_ref", header.state_machine_ref)
        )
        header.unknown_28 = int(header_ir.get("unknown_28", header.unknown_28))
        header.unknown_68 = int(header_ir.get("unknown_68", header.unknown_68))
        header.header_data2 = bytes.fromhex(
            header_ir.get("header_data2_hex", "") or ""
        ).ljust(8, b"\x00")[:8]
        header.padding = bytes.fromhex(
            header_ir.get("padding_hex", "") or ""
        ).ljust(12, b"\x00")[:12]
        header.version = version

        # ---- build every section ------------------------------------- #
        sections: dict[str, bytes] = {}
        for key in ("wwise_callback", "pre_light_list", "transform_info", "customization",
                    "unk_header1", "connecting_bones", "bone_info", "lod_group_list"):
            blob = payloads.get(f"opaque_{key}.bin")
            if blob is None:
                continue
            sections[key] = blob

        light_list = LightList()
        light_list.unknown0 = list(ir.get("lights_unknown0") or [0, 0, 0])
        light_list.lights = [Light.from_json(item) for item in (ir.get("lights") or [])]
        sections["light_list"] = light_list.to_bytes()

        streams = self._streams_from_json(ir, version)
        stream_info_blob = self._build_stream_info(streams)
        if stream_info_blob:
            sections["stream_info"] = stream_info_blob

        section_offsets = ir.get("section_offsets") or {}
        # a section is only emitted when the original had one (or the IR grew
        # content for it) - retail units exist without a materials section
        if (ir.get("meshes") or "mesh_info" in section_offsets):
            sections["mesh_info"] = self._build_mesh_info(ir)
        if (ir.get("materials") or ir.get("has_materials_section")
                or "materials" in section_offsets):
            sections["materials"] = self._build_materials(ir)
        unreversed2 = payloads.get("unreversed2.bin", b"")

        # ---- lay the sections out in the original order -------------- #
        original_order = sorted(
            (
                (int(offset), name)
                for name, offset in (ir.get("section_offsets") or {}).items()
                if offset and name in sections
            ),
            key=lambda item: item[0],
        )
        known = {name for _, name in original_order}
        order = [name for _, name in original_order]
        # sections that exist but had no offset recorded (new content)
        for name in ("wwise_callback", "pre_light_list", "light_list", "lod_group_list",
                     "transform_info", "customization", "unk_header1", "connecting_bones",
                     "bone_info", "stream_info", "mesh_info", "materials"):
            if name in sections and name not in known:
                order.append(name)
        offsets_hint = {
            name: int(offset) for offset, name in original_order
        }

        body = Stream.writer()
        body.seek(HEADER_SIZE)
        new_offsets: dict[str, int] = {}
        for name in order:
            hint = offsets_hint.get(name)
            if hint is not None and hint >= body.tell():
                # reproduce the original layout (including its padding)
                body.seek(hint)
            elif name == "stream_info":
                # new stream info sections start on a 16 byte boundary
                body.seek(align_up(body.tell(), 16))
            new_offsets[name] = body.tell()
            body.write(sections[name])

        body.write(unreversed2)
        if ir.get("has_ending_marker", True):
            header.ending_offset = body.tell()
            body.uint64(int(ir.get("ending_bytes", 0)))
        else:
            # units without the terminator simply end after the last section
            header.ending_offset = 0
        tail = payloads.get("toc_tail.bin", b"")
        if tail:
            body.write(tail)

        header.wwise_callback_offset = new_offsets.get("wwise_callback", 0)
        header.pre_light_list_offset = new_offsets.get("pre_light_list", 0)
        header.light_list_offset = new_offsets.get("light_list", 0)
        header.lod_group_list_offset = new_offsets.get("lod_group_list", 0)
        header.transform_info_offset = new_offsets.get("transform_info", 0)
        header.customization_offset = new_offsets.get("customization", 0)
        header.unk_header1_offset = new_offsets.get("unk_header1", 0)
        header.connecting_bones_offset = new_offsets.get("connecting_bones", 0)
        header.bone_info_offset = new_offsets.get("bone_info", 0)
        header.stream_info_offset = new_offsets.get("stream_info", 0)
        header.mesh_info_offset = new_offsets.get("mesh_info", 0)
        header.materials_offset = new_offsets.get("materials", 0)

        toc = bytearray(body.Data)
        toc[0:HEADER_SIZE] = header.to_bytes()

        gpu = payloads.get("gpu.bin", bytes(entry.gpu_data))
        stream = payloads.get("stream.bin", bytes(entry.stream_data))
        return bytes(toc), bytes(gpu), bytes(stream)

    # ------------------------------------------------------------------ #
    def _streams_from_json(self, ir: dict, version: int) -> list[StreamInfo]:
        result: list[StreamInfo] = []
        for item in ir.get("streams") or []:
            info = StreamInfo()
            info.component_info_id = int(item.get("component_info_id", 0))
            info.num_vertices = int(item.get("num_vertices", 0))
            info.vertex_stride = int(item.get("vertex_stride", 0))
            vertex = item.get("vertex_buffer") or {}
            info.vertex_buffer_id = int(vertex.get("id", 0))
            info.vertex_buffer_unk1 = int(vertex.get("unk1", 0))
            info.vertex_buffer_unk2 = int(vertex.get("unk2", 0))
            info.vertex_buffer_unk3 = int(vertex.get("unk3", 0))
            info.vertex_buffer_offset = int(vertex.get("offset", 0))
            info.vertex_buffer_size = int(vertex.get("size", 0))
            index = item.get("index_buffer") or {}
            info.index_buffer_id = int(index.get("id", 0))
            info.index_buffer_unk1 = int(index.get("unk1", 0))
            info.index_buffer_unk2 = int(index.get("unk2", 0))
            info.index_buffer_unk3 = int(index.get("unk3", 0))
            info.index_buffer_type = int(index.get("type", 0))
            info.num_indices = int(index.get("num_indices", 0))
            info.index_buffer_offset = int(index.get("offset", 0))
            info.index_buffer_size = int(index.get("size", 0))
            info.unk_ending_bytes = bytes.fromhex(
                item.get("unk_ending_bytes_hex", "") or ""
            ).ljust(16, b"\x00")[:16]
            info.component_region_tail = bytes.fromhex(
                item.get("component_region_tail_hex", "") or ""
            )
            components = []
            for comp in item.get("components") or []:
                components.append(
                    {
                        "type": int(comp.get("type", 0)),
                        "format": int(comp.get("format", 0)),
                        "index": int(comp.get("index", 0)),
                        "unknown": int(comp.get("unknown", 0)),
                    }
                )
            info.components = components
            info.num_components_declared = len(components)
            info.unknown_offsets = list(item.get("unknown_offsets") or [])
            info.section_offset = int(item.get("section_offset", 0))
            result.append(info)
        return result

    def _build_stream_info(self, streams: list[StreamInfo]) -> bytes:
        if not streams:
            return b""
        section = Stream.writer()
        section.uint32(len(streams))
        offset_slots = len(streams)
        section.seek(section.tell() + 4 * offset_slots)      # offsets, patched later
        unknown_ids = list(streams[0].unknown_offsets or [])
        for index in range(offset_slots):
            # the engine stores the id of the mesh that owns each stream here
            value = unknown_ids[index] if index < len(unknown_ids) else 0
            section.uint32(value)
        section.uint32(0)                                     # unknown 2

        for info in streams:
            if info.section_offset and info.section_offset >= section.tell():
                section.seek(info.section_offset)
            info.offset = section.tell()
            section.uint64(info.component_info_id)
            for comp in info.components:
                section.uint32(comp["type"])
                section.uint32(comp["format"])
                section.uint32(comp["index"])
                section.uint64(comp["unknown"])
            used = len(info.components) * 20
            if used > StreamInfo.COMPONENT_REGION:
                raise CodecError(
                    f"unit: too many vertex components ({len(info.components)}), "
                    f"the reserved region holds at most "
                    f"{StreamInfo.COMPONENT_REGION // 20}"
                )
            remaining = StreamInfo.COMPONENT_REGION - used
            tail = info.component_region_tail or b"\x00" * remaining
            section.write(tail[:remaining].ljust(remaining, b"\x00"))
            section.uint64(len(info.components))
            section.uint64(info.vertex_buffer_id)
            section.uint64(info.vertex_buffer_unk1)
            section.uint32(info.num_vertices)
            section.uint32(info.vertex_stride)
            section.uint64(info.vertex_buffer_unk2)
            section.uint64(info.vertex_buffer_unk3)
            section.uint64(info.index_buffer_id)
            section.uint64(info.index_buffer_unk1)
            section.uint32(info.num_indices)
            section.uint32(info.index_buffer_type)
            section.uint64(info.index_buffer_unk2)
            section.uint64(info.index_buffer_unk3)
            section.uint32(info.vertex_buffer_offset)
            section.uint32(info.vertex_buffer_size)
            section.uint32(info.index_buffer_offset)
            section.uint32(info.index_buffer_size)
            section.write(info.unk_ending_bytes[:16].ljust(16, b"\x00"))

        # patch the offset table (relative to the section start)
        section.seek(4)
        for info in streams:
            section.uint32(info.offset)
        section.seek(len(section.Data))
        return section.getvalue()

    def _build_mesh_info(self, ir: dict) -> bytes:
        meshes = ir.get("meshes") or []
        section = Stream.writer()
        section.uint32(len(meshes))
        offsets_slot = section.tell()
        section.seek(section.tell() + 4 * len(meshes))
        unknown_ids = list(ir.get("mesh_unknown_ids") or [])
        for index, mesh in enumerate(meshes):
            value = (
                unknown_ids[index]
                if index < len(unknown_ids)
                else int(mesh.get("mesh_id", 0))
            )
            section.uint32(value)
        offsets: list[int] = []
        for mesh in meshes:
            hint = int(mesh.get("section_offset", 0))
            if hint and hint >= section.tell():
                section.seek(hint)
            offsets.append(section.tell())
            unknown = mesh.get("unknown") or {}
            material_ids = [int(v) for v in (mesh.get("material_slot_ids") or [])]
            sections = mesh.get("sections") or []
            material_offset = int(mesh.get("material_offset", 0)) or 0x80
            section.uint64(int(unknown.get("unk1", 0)))
            section.write(bytes.fromhex(unknown.get("unk2_hex", "") or "").ljust(32, b"\x00")[:32])
            section.uint32(int(mesh.get("mesh_id", 0)))
            section.uint32(int(unknown.get("unk3", 0)))
            section.uint32(int(mesh.get("transform_index", 0)))
            section.uint32(int(unknown.get("unk4", 0)))
            section.int32(int(mesh.get("lod_index", -1)))
            section.uint32(int(mesh.get("stream_index", 0)))
            section.write(bytes.fromhex(unknown.get("unk6_hex", "") or "").ljust(40, b"\x00")[:40])
            section.uint32(len(material_ids))
            section.uint32(material_offset)
            section.uint64(int(unknown.get("unk8", 0)))
            section.uint32(len(sections))
            section.uint32(material_offset + 4 * len(material_ids))
            for value in material_ids:
                section.uint32(value)
            for item in sections:
                section.uint32(int(item.get("material_index", 0)))
                section.uint32(int(item.get("vertex_offset", 0)))
                section.uint32(int(item.get("num_vertices", 0)))
                section.uint32(int(item.get("index_offset", 0)))
                section.uint32(int(item.get("num_indices", 0)))
                section.uint32(int(item.get("group_index", 0)))

        section.seek(offsets_slot)
        for offset in offsets:
            section.uint32(offset)
        section.seek(len(section.Data))
        return section.getvalue()

    def _build_materials(self, ir: dict) -> bytes:
        materials = ir.get("materials") or []
        section = Stream.writer()
        section.uint32(len(materials))
        for item in materials:
            section.uint32(int(item.get("slot_id", 0)))
        for item in materials:
            section.uint64(int(item.get("material_id", 0)))
        return section.getvalue()


def _parse_transform_view(data: bytes, offset: int, names: NameTable) -> list[dict]:
    """Read-only view of the transform info section (bone name hashes!)."""
    try:
        stream = Stream.reader(data)
        stream.seek(offset)
        count = stream.uint32()
        stream.read(12)
        if count <= 0 or count > 100000:
            return []
        transforms = []
        for index in range(count):
            rotation = [stream.float32() for _ in range(9)]
            position = [stream.float32() for _ in range(3)]
            scale = [stream.float32() for _ in range(3)]
            stream.float32()  # alignment dummy
            transforms.append(
                {
                    "index": index,
                    "rotation": rotation,
                    "position": position,
                    "scale": scale,
                }
            )
        matrices = [[stream.float32() for _ in range(16)] for _ in range(count)]
        entries = [
            {"increment": stream.uint16(), "parent_bone": stream.uint16()}
            for _ in range(count)
        ]
        hashes = [stream.uint32() for _ in range(count)]
        for index, item in enumerate(transforms):
            item["name_hash"] = str(hashes[index])
            item["name"] = names.bone(hashes[index])
            item["parent_bone"] = entries[index]["parent_bone"]
            item["increment"] = entries[index]["increment"]
            item["matrix"] = matrices[index]
        return transforms
    except Exception:  # noqa: BLE001 - the section is opaque, a view may fail
        return []


def _first_set(*values: int) -> int:
    """First non zero value (kept for the unit tests / older call sites)."""
    for value in values:
        if value:
            return value
    raise FormatError("unit: no following section offset found")


def _parse_customization(data: bytes) -> dict:
    """Best effort view of the customization section (read only)."""
    try:
        cursor = 24
        result = {}
        for key in ("body_type", "slot", "weight", "piece_type"):
            length = int.from_bytes(data[cursor:cursor + 4], "little")
            cursor += 4
            raw = data[cursor:cursor + length]
            cursor += length
            result[key] = raw.replace(b"\x00", b"").decode("utf-8", "replace")
            cursor += 12
        return result
    except Exception:  # noqa: BLE001 - this section is not understood
        return {"error": "could not parse customization section"}


def _decode_component(raw: bytes, format_id: int, version: int) -> Any:
    table = _format_table(version)
    name = table.get(format_id, ("", 0))[0]
    if not raw:
        return None
    if name == "float":
        return struct.unpack("<f", raw[:4])[0]
    if name == "vec2_float":
        return list(struct.unpack("<2f", raw[:8]))
    if name == "vec3_float":
        return list(struct.unpack("<3f", raw[:12]))
    if name == "vec4_float":
        return list(struct.unpack("<4f", raw[:16]))
    if name == "rgba_r8g8b8a8":
        return [value / 255.0 for value in raw[:4]]
    if name == "vec4_uint8":
        return list(raw[:4])
    if name == "vec2_uint8":
        return list(raw[:2])
    if name == "vec3_uint8":
        return list(raw[:3])
    if name == "uint8":
        return raw[0]
    if name == "vec4_uint32":
        return list(struct.unpack("<4I", raw[:16]))
    if name == "vec2_uint32":
        return list(struct.unpack("<2I", raw[:8]))
    if name == "vec3_uint32":
        return list(struct.unpack("<3I", raw[:12]))
    if name == "uint32":
        return struct.unpack("<I", raw[:4])[0]
    if name == "vec4_half":
        return list(struct.unpack("<4e", raw[:8]))
    if name == "vec3_half":
        return list(struct.unpack("<3e", raw[:6]))
    if name == "vec2_half":
        return list(struct.unpack("<2e", raw[:4]))
    if name == "float16":
        return struct.unpack("<e", raw[:2])[0]
    if name == "vec4_1010102":
        return _ten_bit_unsigned(struct.unpack("<I", raw[:4])[0])
    if name == "unk_normal":
        return list(decode_packed_oct_normal(struct.unpack("<I", raw[:4])[0]))
    return raw.hex()


def _json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, indent=1).encode("utf-8")


codec = register(UnitCodec())
