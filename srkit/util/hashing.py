"""Stingray name hashing (MurmurHash64A, plus the 32 bit variant)."""

from __future__ import annotations

__all__ = ["murmur64_hash", "murmur32_hash", "hash_name"]

_MASK = 2 ** 64 - 1


def murmur64_hash(data: bytes | bytearray | str, seed: int = 0) -> int:
    """MurmurHash64A, matching the engine's ``murmur_hash_64``."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    data = bytes(data)
    m = 0xC6A4A7935BD1E995
    r = 47

    h = (seed ^ ((m * len(data)) & _MASK)) & _MASK
    off = (len(data) // 8) * 8
    for pos in range(0, off, 8):
        k = int.from_bytes(data[pos:pos + 8], "little")
        k = (k * m) & _MASK
        k ^= k >> r
        k = (k * m) & _MASK
        h ^= k
        h = (h * m) & _MASK

    tail = len(data) & 7
    if tail >= 7:
        h ^= data[off + 6] << 48
    if tail >= 6:
        h ^= data[off + 5] << 40
    if tail >= 5:
        h ^= data[off + 4] << 32
    if tail >= 4:
        h ^= data[off + 3] << 24
    if tail >= 3:
        h ^= data[off + 2] << 16
    if tail >= 2:
        h ^= data[off + 1] << 8
    if tail >= 1:
        h ^= data[off]
        h = (h * m) & _MASK

    h ^= h >> r
    h = (h * m) & _MASK
    h ^= h >> r
    return h


def murmur32_hash(data: bytes | bytearray | str, seed: int = 0) -> int:
    """The 32 bit value the engine uses for bone / shader-variable names."""
    return murmur64_hash(data, seed) >> 32


def hash_name(name: str | int) -> int:
    """Mirror of ``compute_bone_name_hash``: numeric strings pass through."""
    if isinstance(name, int):
        return name
    try:
        return int(name)
    except ValueError:
        return murmur32_hash(name.encode("utf-8"))


def djb2_type_id(name: str | bytes) -> int:
    """The datalibrary (``LDLD``) type id of a type name.

    Measured on build 25480438: the value is the **standard djb2 with the usual 5381 seed, minus
    5381** (masked to 32 bits).  Verified against ten published ids, e.g.
    ``djb2("ProjectileSettings") == 0xBD4057C7`` and ``0xBD4057C7 - 0x1505 == 0xBD4042C2`` which is
    the id the engine actually stores in the block header.  Note that a seed-0 recurrence is *not*
    equivalent -- the recurrence is not offset-invariant, so the subtraction must happen last.
    """
    data = name.encode("utf-8") if isinstance(name, str) else bytes(name)
    value = 5381
    for byte in data:
        value = (value * 33 + byte) & 0xFFFFFFFF
    return (value - 5381) & 0xFFFFFFFF
