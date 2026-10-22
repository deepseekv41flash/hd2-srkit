"""FileID (u64) and SlotID (u32) rewriting inside Unit / Material blobs.

Only the fields where Stingray stores cross-resource references are touched;
geometry and texture pixel data are copied byte for byte.

Ported from ``eigeen/hd2-mod-crates`` (``crates/hd2-migrator-core/src/refs.rs``
and ``crates/hd2-archive-format/src/refs.rs``, MIT OR Apache-2.0).

Unit blob layout (offsets inside the toc blob)::

    0x00 u64 UnkRef1
    0x08 u64 BonesRef
    0x10 u64 CompositeRef
    0x18 u64 UnkRef2
    0x20 u64 StateMachineRef
    0x70 u32 MaterialsOffset
      @MaterialsOffset:
        u32 NumMaterials
        u32[NumMaterials] SectionIDs   (slot ids, also remapped)
        u64[NumMaterials] MaterialIDs  (FileIDs, remapped)

Material blob::

    0x40 u32 NumTextures
        u32[NumTextures] TexUnks
        u64[NumTextures] TexIDs        (FileIDs, remapped)
"""

from __future__ import annotations

import struct

from ..util.ids import TYPE_MATERIAL, TYPE_UNIT

__all__ = [
    "UNIT_HEADER_REFS",
    "rewrite_u64_refs",
    "rewrite_unit",
    "rewrite_material",
    "rewrite",
    "list_unit_refs",
    "list_material_refs",
]

UNIT_HEADER_REFS = (0x00, 0x08, 0x10, 0x18, 0x20)
UNIT_MATERIALS_OFFSET_FIELD = 0x70
UNIT_MIN_LEN = 0x80
MATERIAL_MIN_LEN = 0x88
MATERIAL_NUM_TEXTURES_FIELD = 0x40
MATERIAL_TEX_BASE = 136


def rewrite_u64_refs(data: bytes, remap: dict[int, int]) -> tuple[bytes, int]:
    """Replace every aligned u64 that appears in ``remap``. Returns (bytes, count)."""
    buf = bytearray(data)
    changed = 0
    for position in range(0, len(buf) - 7, 8):
        old = struct.unpack_from("<Q", buf, position)[0]
        new = remap.get(old)
        if new is None or new == old:
            continue
        struct.pack_into("<Q", buf, position, new)
        changed += 1
    return bytes(buf), changed


def rewrite_unit(
    toc_data: bytes,
    remap: dict[int, int] | None = None,
    slot_remap: dict[int, int] | None = None,
) -> bytes:
    """Rewrite the FileID references (and optionally slot ids) of a unit blob."""
    if len(toc_data) < UNIT_MIN_LEN:
        return bytes(toc_data)
    buf = bytearray(toc_data)
    table = remap or {}

    for offset in UNIT_HEADER_REFS:
        old = struct.unpack_from("<Q", buf, offset)[0]
        new = table.get(old)
        if new is not None and new != old:
            struct.pack_into("<Q", buf, offset, new)

    if len(buf) >= UNIT_MATERIALS_OFFSET_FIELD + 4:
        materials_offset = struct.unpack_from("<I", buf, UNIT_MATERIALS_OFFSET_FIELD)[0]
        if materials_offset and materials_offset + 4 <= len(buf):
            count = struct.unpack_from("<I", buf, materials_offset)[0]
            ids_start = materials_offset + 4 + 4 * count
            if ids_start + 8 * count <= len(buf):
                for index in range(count):
                    position = ids_start + 8 * index
                    old = struct.unpack_from("<Q", buf, position)[0]
                    new = table.get(old)
                    if new is not None and new != old:
                        struct.pack_into("<Q", buf, position, new)

    if slot_remap:
        for position in range(0, len(buf) - 3, 4):
            old = struct.unpack_from("<I", buf, position)[0]
            new = slot_remap.get(old)
            if new is not None and new != old:
                struct.pack_into("<I", buf, position, new)

    return bytes(buf)


def rewrite_material(toc_data: bytes, remap: dict[int, int] | None = None) -> bytes:
    """Rewrite the texture FileID array of a material blob."""
    if len(toc_data) < MATERIAL_MIN_LEN:
        return bytes(toc_data)
    buf = bytearray(toc_data)
    table = remap or {}
    count = struct.unpack_from("<I", buf, MATERIAL_NUM_TEXTURES_FIELD)[0]
    ids_start = MATERIAL_TEX_BASE + 4 * count
    if count == 0 or ids_start + 8 * count > len(buf):
        return bytes(buf)
    for index in range(count):
        position = ids_start + 8 * index
        old = struct.unpack_from("<Q", buf, position)[0]
        new = table.get(old)
        if new is not None and new != old:
            struct.pack_into("<Q", buf, position, new)
    return bytes(buf)


def rewrite(
    type_id: int,
    toc_data: bytes,
    remap: dict[int, int] | None = None,
    slot_remap: dict[int, int] | None = None,
) -> bytes:
    """Dispatch on the resource type; unknown types are returned unchanged."""
    if type_id == TYPE_UNIT:
        return rewrite_unit(toc_data, remap, slot_remap)
    if type_id == TYPE_MATERIAL:
        return rewrite_material(toc_data, remap)
    return bytes(toc_data)


def list_unit_refs(toc_data: bytes) -> list[int]:
    """Every FileID a unit blob points at (5 header refs, then MaterialIDs)."""
    if len(toc_data) < UNIT_MIN_LEN:
        return []
    refs = [struct.unpack_from("<Q", toc_data, offset)[0] for offset in UNIT_HEADER_REFS]
    materials_offset = struct.unpack_from("<I", toc_data, UNIT_MATERIALS_OFFSET_FIELD)[0]
    if materials_offset and materials_offset + 4 <= len(toc_data):
        count = struct.unpack_from("<I", toc_data, materials_offset)[0]
        ids_start = materials_offset + 4 + 4 * count
        if ids_start + 8 * count <= len(toc_data):
            refs.extend(struct.unpack_from("<Q", toc_data, ids_start + 8 * index)[0]
                        for index in range(count))
    return refs


def list_material_refs(toc_data: bytes) -> list[int]:
    """Every texture FileID a material blob points at."""
    if len(toc_data) < MATERIAL_MIN_LEN:
        return []
    count = struct.unpack_from("<I", toc_data, MATERIAL_NUM_TEXTURES_FIELD)[0]
    ids_start = MATERIAL_TEX_BASE + 4 * count
    if count == 0 or ids_start + 8 * count > len(toc_data):
        return []
    return [struct.unpack_from("<Q", toc_data, ids_start + 8 * index)[0]
            for index in range(count)]
