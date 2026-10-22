"""Empty-unit synthesis: fill patch slots with a mesh that draws nothing.

HD2's patch system overrides resources **by FileID**: if a patch archive carries
FileID X, the game uses the patch's copy, otherwise the stock resource.  An
appearance mod that covers only some of a target's Unit slots therefore leaves
the rest showing stock parts.  The fix is to add an *empty* unit for every
uncovered slot, which is what this module builds.

An "empty unit" is a real unit resource whose material references and index
counts have been zeroed, so it parses and loads but renders nothing.

Ported from ``eigeen/hd2-mod-crates`` (``crates/hd2-migrator-core/src/padding.rs``
and its ``empty_mesh`` asset handling, MIT OR Apache-2.0).  The template itself
is not vendored: extract one from any patch that carries a small unit
(:func:`extract_template`) or build one explicitly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..errors import FormatError
from ..util.ids import TYPE_UNIT
from .package import Package, TocEntry
from .unit_refs import rewrite_unit

__all__ = [
    "EmptyUnitAudit",
    "EmptyUnitTemplate",
    "audit_empty_unit",
    "sanitize_empty_unit",
    "find_empty_unit_candidates",
    "extract_template",
    "pad_patch",
    "PADDING_MODES",
]

PADDING_MODES = ("disabled", "sanitized", "verbatim")

UNIT_HEADER_REFS = (0x00, 0x08, 0x10, 0x18, 0x20)
UNIT_MIN_HEADER = 0x74
UNIT_MATERIALS_OFFSET_FIELD = 0x70
STREAM_INFO_SIZE = 416
MESH_INFO_SIZE = 128
SECTION_SIZE = 24


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #
def _header_offsets(toc_data: bytes) -> tuple[int, int, int]:
    """(stream_info, mesh_info, materials) offsets, or zeros when unknown."""
    if len(toc_data) < UNIT_MIN_HEADER:
        return 0, 0, 0
    stream_off = struct.unpack_from("<I", toc_data, 0x5C)[0]
    mesh_off = struct.unpack_from("<I", toc_data, 0x64)[0]
    materials_off = struct.unpack_from("<I", toc_data, 0x70)[0]
    return stream_off, mesh_off, materials_off


def _read_material_ids(toc_data: bytes) -> list[int]:
    _stream, _mesh, materials_off = _header_offsets(toc_data)
    if not materials_off or materials_off + 4 > len(toc_data):
        return []
    count = struct.unpack_from("<I", toc_data, materials_off)[0]
    ids_start = materials_off + 4 + 4 * count
    if ids_start + 8 * count > len(toc_data):
        return []
    return [struct.unpack_from("<Q", toc_data, ids_start + 8 * i)[0] for i in range(count)]


def _stream_info_bases(toc_data: bytes) -> list[int]:
    stream_off, _mesh, _materials = _header_offsets(toc_data)
    if not stream_off or stream_off + 4 > len(toc_data):
        return []
    count = struct.unpack_from("<I", toc_data, stream_off)[0]
    bases = []
    for index in range(count):
        position = stream_off + 4 + 4 * index
        if position + 4 > len(toc_data):
            break
        relative = struct.unpack_from("<I", toc_data, position)[0]
        base = stream_off + relative
        if base + STREAM_INFO_SIZE <= len(toc_data):
            bases.append(base)
    return bases


def _mesh_info_bases(toc_data: bytes) -> list[int]:
    _stream, mesh_off, _materials = _header_offsets(toc_data)
    if not mesh_off or mesh_off + 4 > len(toc_data):
        return []
    count = struct.unpack_from("<I", toc_data, mesh_off)[0]
    bases = []
    for index in range(count):
        position = mesh_off + 4 + 4 * index
        if position + 4 > len(toc_data):
            break
        relative = struct.unpack_from("<I", toc_data, position)[0]
        base = mesh_off + relative
        if base + MESH_INFO_SIZE <= len(toc_data):
            bases.append(base)
    return bases


def _section_offsets(toc_data: bytes) -> list[int]:
    offsets = []
    for base in _mesh_info_bases(toc_data):
        num_sections = struct.unpack_from("<I", toc_data, base + 120)[0]
        section_relative = struct.unpack_from("<I", toc_data, base + 124)[0]
        for index in range(num_sections):
            offset = base + section_relative + SECTION_SIZE * index
            if offset + SECTION_SIZE <= len(toc_data):
                offsets.append(offset)
    return offsets


# --------------------------------------------------------------------------- #
# audit / sanitize
# --------------------------------------------------------------------------- #
@dataclass
class EmptyUnitAudit:
    """What a unit blob still points at and how much it would draw."""

    header_refs: list[int] = field(default_factory=list)
    material_ids: list[int] = field(default_factory=list)
    section_indices: list[int] = field(default_factory=list)

    @property
    def num_indices(self) -> int:
        return sum(self.section_indices)

    def external_refs(self) -> list[int]:
        refs = [value for value in self.header_refs if value]
        refs.extend(value for value in self.material_ids if value)
        return refs

    def is_dependency_free(self) -> bool:
        return not self.external_refs()

    def is_non_drawing(self) -> bool:
        return self.num_indices == 0 and all(value == 0 for value in self.section_indices)

    def describe(self) -> str:
        return (
            f"refs={len(self.external_refs())} materials={len(self.material_ids)} "
            f"indices={self.num_indices} sections={len(self.section_indices)} "
            f"dependency_free={self.is_dependency_free()} "
            f"non_drawing={self.is_non_drawing()}"
        )


def audit_empty_unit(toc_data: bytes) -> EmptyUnitAudit:
    """Report the cross-resource refs and index counts of a unit blob."""
    header_refs = []
    if len(toc_data) >= 0x28:
        header_refs = [struct.unpack_from("<Q", toc_data, offset)[0] for offset in UNIT_HEADER_REFS]
    section_indices = []
    for offset in _section_offsets(toc_data):
        section_indices.append(struct.unpack_from("<I", toc_data, offset + 16)[0])
    return EmptyUnitAudit(
        header_refs=header_refs,
        material_ids=_read_material_ids(toc_data),
        section_indices=section_indices,
    )


def sanitize_empty_unit(toc_data: bytes) -> bytes:
    """Zero every material reference and index count so the unit draws nothing."""
    buf = bytearray(toc_data)

    ids = _read_material_ids(bytes(buf))
    _stream, _mesh, materials_off = _header_offsets(bytes(buf))
    if materials_off and ids:
        ids_start = materials_off + 4 + 4 * len(ids)
        for index in range(len(ids)):
            position = ids_start + 8 * index
            if position + 8 <= len(buf):
                struct.pack_into("<Q", buf, position, 0)

    for base in _stream_info_bases(bytes(buf)):
        struct.pack_into("<I", buf, base + 384, 0)
        struct.pack_into("<I", buf, base + 412, 0)

    for offset in _section_offsets(bytes(buf)):
        struct.pack_into("<I", buf, offset + 16, 0)

    return bytes(buf)


# --------------------------------------------------------------------------- #
# template
# --------------------------------------------------------------------------- #
def _scan_u32_words(data: bytes) -> list[int]:
    return [struct.unpack_from("<I", data, position)[0]
            for position in range(0, len(data) - 3, 4)]


@dataclass
class EmptyUnitTemplate:
    """A unit resource that can be cloned onto any FileID as an empty slot."""

    name: str
    toc_data: bytes
    gpu_data: bytes
    stream_data: bytes
    source_file_id: int
    source_slot_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_parts(cls, name: str, toc_data: bytes, gpu_data: bytes = b"",
                   stream_data: bytes = b"", source_file_id: int = 0) -> "EmptyUnitTemplate":
        if not toc_data:
            raise FormatError(f"{name}: empty unit template has no toc data")
        return cls(
            name=name,
            toc_data=bytes(toc_data),
            gpu_data=bytes(gpu_data),
            stream_data=bytes(stream_data),
            source_file_id=source_file_id,
            source_slot_ids=_scan_u32_words(toc_data),
        )

    def audit(self) -> EmptyUnitAudit:
        return audit_empty_unit(self.toc_data)

    def clone_for(
        self,
        target_file_id: int,
        slot_id_remap: dict[int, int] | None = None,
        file_id_remap: dict[int, int] | None = None,
        verbatim: bool = False,
    ) -> TocEntry:
        """Build a unit entry for ``target_file_id`` from this template.

        ``verbatim`` keeps the bytes exactly as extracted; otherwise the blob is
        sanitized (no material refs, no indices) and the references are remapped.
        """
        if verbatim:
            toc = self.toc_data
        else:
            toc = sanitize_empty_unit(self.toc_data)
            toc = rewrite_unit(toc, file_id_remap or {}, slot_id_remap)
        entry = TocEntry(file_id=int(target_file_id), type_id=TYPE_UNIT)
        entry.set_data(toc=toc, gpu=self.gpu_data, stream=self.stream_data)
        return entry


def find_empty_unit_candidates(package: Package) -> list[TocEntry]:
    """Unit entries ordered by (gpu size, toc size) — smallest is likeliest empty."""
    units = package.entries_of_type(TYPE_UNIT)
    return sorted(units, key=lambda entry: (len(entry.gpu_data), len(entry.toc_data)))


def extract_template(package: Package, name: str = "", file_id: int | None = None) -> EmptyUnitTemplate:
    """Take a unit of ``package`` as an empty-unit template.

    Without ``file_id`` the smallest unit (by gpu, then toc size) is used; the
    caller can name an exact resource when the heuristic picks a degenerate one.
    """
    if file_id is not None:
        chosen = package.entry(int(file_id), TYPE_UNIT)
        if chosen is None:
            raise FormatError(
                f"{name or package.name or 'package'}: no Unit entry with file id "
                f"0x{int(file_id):016X}"
            )
    else:
        candidates = find_empty_unit_candidates(package)
        if not candidates:
            raise FormatError(
                f"{name or package.name or 'package'}: no Unit entries to use as a template"
            )
        chosen = candidates[0]
    return EmptyUnitTemplate.from_parts(
        name=package.name or name or "<template>",
        toc_data=bytes(chosen.toc_data),
        gpu_data=bytes(chosen.gpu_data),
        stream_data=bytes(chosen.stream_data),
        source_file_id=chosen.file_id,
    )


def pad_patch(
    package: Package,
    target_slots: list[int],
    template: EmptyUnitTemplate,
    mode: str = "sanitized",
    slot_id_remap: dict[int, int] | None = None,
) -> list[int]:
    """Add an empty unit for every target slot the patch does not cover.

    Returns the FileIDs that were added.  ``mode`` is one of
    :data:`PADDING_MODES`; ``"disabled"`` adds nothing.
    """
    if mode not in PADDING_MODES:
        raise FormatError(f"unknown padding mode {mode!r}; expected one of {PADDING_MODES}")
    if mode == "disabled":
        return []
    covered = {entry.file_id for entry in package.entries_of_type(TYPE_UNIT)}
    extras = [slot for slot in target_slots if slot not in covered]
    for slot in extras:
        entry = template.clone_for(slot, slot_id_remap=slot_id_remap,
                                   verbatim=(mode == "verbatim"))
        package.add_entry(entry)
    return extras
