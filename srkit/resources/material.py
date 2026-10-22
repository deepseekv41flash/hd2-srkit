"""``material`` resource codec.

Layout (``stingray/material.py`` in the add-on)::

    0x00  12 bytes unknown
    0x0C  u32 end offset (file size)
    0x10  u64 unknown2
    0x18  u64 parent material id
    0x20  32 bytes unknown
    0x40  u32 texture count
    0x44  36 bytes unknown
    0x68  u32 shader variable count
    0x6C  12 bytes unknown
    0x78  u32 variable data size
    0x7C  12 bytes unknown
    0x88  u32 texture unknowns[texture count]
    ...   u64 texture ids[texture count]
    ...   shader variable table, 20 bytes each
            u32 class (0 scalar, 1 vec2, 2 vec3, 3 vec4, 12 other)
            u32 element count
            u32 name hash (see hashlists/shadervariables.txt)
            u32 offset into the value blob
            u32 element stride
    ...   value blob (``variable data size`` bytes), variables read floats at
          ``offset`` relative to the start of the blob
"""

from __future__ import annotations

import struct

from ..container.package import TocEntry
from ..errors import CodecError
from ..util.binary import Stream
from ..util.ids import PARENT_MATERIALS, NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["MaterialCodec", "codec", "SHADER_VARIABLE_CLASSES"]

SHADER_VARIABLE_CLASSES = {
    0: "Scalar",
    1: "Vector2",
    2: "Vector3",
    3: "Vector4",
    12: "Other",
}


def _hex(value: bytes) -> str:
    return bytes(value).hex()


class MaterialCodec(ResourceCodec):
    name = "material"
    type_ids: tuple[int, ...] = (0xEAC0B497876ADEDF,)  # TYPE_MATERIAL

    def decode(self, entry: TocEntry, names: NameTable, options=None) -> DecodedResource:
        data = bytes(entry.toc_data)
        stream = Stream.reader(data)
        undat1 = bytes(stream.bytes(b"", 12))
        end_offset = stream.uint32()
        undat2 = stream.uint64()
        parent_id = stream.uint64()
        undat3 = bytes(stream.bytes(b"", 32))
        num_textures = stream.uint32()
        undat4 = bytes(stream.bytes(b"", 36))
        num_variables = stream.uint32()
        undat5 = bytes(stream.bytes(b"", 12))
        variable_data_size = stream.uint32()
        undat6 = bytes(stream.bytes(b"", 12))

        tex_unks = [stream.uint32() for _ in range(num_textures)]
        tex_ids = [stream.uint64() for _ in range(num_textures)]

        variables = []
        for _ in range(num_variables):
            variables.append(
                {
                    "class": stream.uint32(),
                    "elements": stream.uint32(),
                    "id": stream.uint32(),
                    "offset": stream.uint32(),
                    "element_stride": stream.uint32(),
                }
            )

        value_start = stream.tell()
        value_blob = data[value_start:value_start + variable_data_size]
        if len(value_blob) < variable_data_size:
            value_blob = value_blob.ljust(variable_data_size, b"\x00")
        leftover = data[value_start + variable_data_size:]

        for item in variables:
            klass = item["class"]
            item["class_name"] = SHADER_VARIABLE_CLASSES.get(klass, f"Unknown({klass})")
            item["name"] = names.shader_variable(item["id"])
            count = klass + 1
            base = item["offset"]
            if base + count * 4 <= len(value_blob):
                item["values"] = list(
                    struct.unpack_from(f"<{count}f", value_blob, base)
                )
                # exact bytes, so an unedited material reproduces bit for bit
                # (float -> JSON -> float is not lossless for NaN payloads)
                item["values_hex"] = value_blob[base:base + count * 4].hex()
            else:
                item["values"] = []
                item["values_hex"] = ""
                item["values_error"] = "offset outside of the value blob"

        ir = self.base_ir(entry, "material")
        ir.update(
            {
                "parent_material_id": str(parent_id),
                "parent_material_template": PARENT_MATERIALS.get(parent_id, ""),
                "end_offset": end_offset,
                "variable_data_size": variable_data_size,
                "textures": [
                    {
                        "unknown": tek,
                        "id": str(tid),
                        "name": names.label(tid) if tid else "",
                    }
                    for tek, tid in zip(tex_unks, tex_ids)
                ],
                "shader_variables": variables,
                "raw_header": {
                    "undat1": _hex(undat1),
                    "undat2": undat2,
                    "undat3": _hex(undat3),
                    "undat4": _hex(undat4),
                    "undat5": _hex(undat5),
                    "undat6": _hex(undat6),
                },
                "leftover_bytes": len(leftover),
            }
        )
        payloads = [
            Payload(
                "values.bin",
                value_blob,
                editable=False,
                note="shader variable value blob (patched from the JSON values)",
            )
        ]
        if leftover:
            payloads.append(self.master_payload("tail.bin", leftover))
        return DecodedResource(ir, payloads)

    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        header = ir.get("raw_header") or {}
        textures = ir.get("textures") or []
        variables = ir.get("shader_variables") or []
        value_blob = bytearray(payloads.get("values.bin", b""))
        declared = int(ir.get("variable_data_size", 0))

        if len(value_blob) < declared:
            value_blob.extend(bytes(declared - len(value_blob)))

        # patch the JSON values back into the blob
        for item in variables:
            values = item.get("values") or []
            if not values:
                continue
            offset = int(item.get("offset", 0))
            packed = struct.pack(f"<{len(values)}f", *[float(v) for v in values])
            if packed.hex() == (item.get("values_hex") or ""):
                continue  # unchanged: keep the original bytes untouched
            end = offset + len(packed)
            if end > len(value_blob):
                raise CodecError(
                    f"material: shader variable at offset {offset} does not fit "
                    f"the {len(value_blob)} byte value blob"
                )
            value_blob[offset:end] = packed

        stream = Stream.writer()
        stream.write(bytes.fromhex(header.get("undat1", "") or "").ljust(12, b"\x00")[:12])
        stream.uint32(int(ir.get("end_offset", 0)))
        stream.uint64(int(header.get("undat2", 0)))
        stream.uint64(int(ir.get("parent_material_id", 0)))
        stream.write(bytes.fromhex(header.get("undat3", "") or "").ljust(32, b"\x00")[:32])
        stream.uint32(len(textures))
        stream.write(bytes.fromhex(header.get("undat4", "") or "").ljust(36, b"\x00")[:36])
        stream.uint32(len(variables))
        stream.write(bytes.fromhex(header.get("undat5", "") or "").ljust(12, b"\x00")[:12])
        stream.uint32(len(value_blob))
        stream.write(bytes.fromhex(header.get("undat6", "") or "").ljust(12, b"\x00")[:12])

        for item in textures:
            stream.uint32(int(item.get("unknown", 0)))
        for item in textures:
            stream.uint64(int(item.get("id", 0)))
        for item in variables:
            stream.uint32(int(item.get("class", 0)))
            stream.uint32(int(item.get("elements", 0)))
            stream.uint32(int(item.get("id", 0)))
            stream.uint32(int(item.get("offset", 0)))
            stream.uint32(int(item.get("element_stride", 0)))
        stream.write(value_blob)
        tail = payloads.get("tail.bin", b"")
        if tail:
            stream.write(tail)
        return stream.getvalue(), b"", b""


codec = register(MaterialCodec())
