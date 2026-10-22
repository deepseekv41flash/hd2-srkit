"""Pure python LZ4 *block* codec (no frame, no dependencies).

Why not just use the ``lz4`` wheel that ships with the Blender add-on?
Because the add-on only ships compiled ``_block``/``_frame`` extensions for
CPython 3.10/3.11, and a standalone tool must run on any interpreter (this
machine has 3.13, where those ``.pyd`` files cannot be loaded).  The format
logic below is self contained and dependency free.

The Stingray ``DSAR`` container stores every chunk as a raw LZ4 block
(``compression_type == 3``), so block level support is all that is needed.

Reference: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
"""

from __future__ import annotations

import os

__all__ = ["compress", "decompress", "native", "LZ4BlockError", "MAX_INPUT_SIZE", "MIN_MATCH"]

MIN_MATCH = 4
MAX_INPUT_SIZE = 0x7E000000
_LAST_LITERALS = 5  # reference compressor keeps the last 5 bytes as literals
_MFLIMIT = 12  # matches may not start closer than 12 bytes to the end


class LZ4BlockError(ValueError):
    """Malformed LZ4 block."""


# --------------------------------------------------------------------------- #
# optional C acceleration
# --------------------------------------------------------------------------- #
#: The pure python decoder runs at ~40 MB/s; the ``lz4`` extension at ~3 GB/s.
#: A full HD2 ``toc`` pass (6.7 GB) is 156 s vs 2 s, so every reader benefits.
#: Only *decompression* is delegated: the pure python compressor stays in
#: charge of writing so that repacked bytes remain bit identical.
_NATIVE_BLOCK = None
_NATIVE_PROBED = False


def native() -> object | None:
    """Return the ``lz4.block`` module, or ``None`` when it is unavailable.

    ``SRKIT_LZ4_PURE=1`` forces the pure python path (used by the selftest to
    keep the fallback decoder covered).
    """
    global _NATIVE_BLOCK, _NATIVE_PROBED
    if not _NATIVE_PROBED:
        _NATIVE_PROBED = True
        if os.environ.get("SRKIT_LZ4_PURE", "") not in ("", "0"):
            _NATIVE_BLOCK = None
        else:
            try:
                import lz4.block as _block

                # smoke test: a one literal block, so a broken wheel is caught here
                if _block.decompress(b"\x10a", uncompressed_size=1) == b"a":
                    _NATIVE_BLOCK = _block
            except Exception:                     # noqa: BLE001 - acceleration is optional
                _NATIVE_BLOCK = None
    return _NATIVE_BLOCK


# --------------------------------------------------------------------------- #
# decompression
# --------------------------------------------------------------------------- #
def decompress(src: bytes, uncompressed_size: int | None = None) -> bytes:
    """Decompress an LZ4 block.

    ``uncompressed_size`` is optional; when given it is used as a sanity check
    (the Stingray chunk table always provides it).  Without it the block carries
    no size information, so the pure python decoder has to be used.

    The C decoder enforces the "last 5 bytes are literals" rule of the LZ4
    block format, which the tolerant python decoder does not, so any refusal
    falls back instead of failing: a block the python decoder can read always
    reads, only the speed differs.
    """
    if src and uncompressed_size:
        block = native()
        if block is not None:
            try:
                out = block.decompress(bytes(src), uncompressed_size=int(uncompressed_size))
            except Exception:                     # noqa: BLE001 - fall back, never fail here
                out = None
            if out is not None and len(out) == uncompressed_size:
                return out
    return _decompress_python(src, uncompressed_size)


def _decompress_python(src: bytes, uncompressed_size: int | None = None) -> bytes:
    if not src:
        if uncompressed_size:
            raise LZ4BlockError("empty block but uncompressed size > 0")
        return b""

    out = bytearray()
    i = 0
    n = len(src)

    while i < n:
        token = src[i]
        i += 1

        # ---- literal run ----
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                if i >= n:
                    raise LZ4BlockError("truncated literal length")
                extra = src[i]
                i += 1
                lit_len += extra
                if extra != 255:
                    break
        if lit_len:
            end = i + lit_len
            if end > n:
                raise LZ4BlockError("truncated literal run")
            out += src[i:end]
            i = end

        if i >= n:
            break  # last sequence: literals only

        # ---- match copy ----
        if i + 2 > n:
            raise LZ4BlockError("truncated match offset")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0:
            raise LZ4BlockError("invalid match offset 0")

        match_len = token & 0x0F
        if match_len == 15:
            while True:
                if i >= n:
                    raise LZ4BlockError("truncated match length")
                extra = src[i]
                i += 1
                match_len += extra
                if extra != 255:
                    break
        match_len += MIN_MATCH

        start = len(out) - offset
        if start < 0:
            raise LZ4BlockError("match offset points before start of output")

        if offset >= match_len:
            out += out[start:start + match_len]
        else:
            # overlapping copy: byte by byte (run length encoding effect)
            for _ in range(match_len):
                out.append(out[start])
                start += 1

    result = bytes(out)
    if uncompressed_size is not None and len(result) != uncompressed_size:
        raise LZ4BlockError(
            f"size mismatch: decoded {len(result)} bytes, expected {uncompressed_size}"
        )
    return result


# --------------------------------------------------------------------------- #
# compression (greedy, hash table on 4 byte sequences)
# --------------------------------------------------------------------------- #
def compress(src: bytes, acceleration: int = 1) -> bytes:
    """Compress ``src`` into a single LZ4 block.

    Greedy matching with a 4 byte hash table; the produced block is valid but
    obviously not identical to the bytes the retail game shipped (compression
    output is not canonical).  Use :func:`srkit.container.dsar` round trips to
    validate instead of raw byte comparison.
    """
    n = len(src)
    if n == 0:
        return b""
    if n > MAX_INPUT_SIZE:
        raise LZ4BlockError("input too large for a single LZ4 block")

    out = bytearray()
    table: dict[bytes, int] = {}
    anchor = 0
    i = 0
    limit = n - _MFLIMIT  # last position where a match may start

    while i < limit:
        seq = src[i:i + MIN_MATCH]
        candidate = table.get(seq)
        table[seq] = i
        if candidate is None or (i - candidate) > 0xFFFF:
            i += acceleration
            continue

        # extend the match forwards
        m = candidate + MIN_MATCH
        j = i + MIN_MATCH
        while j < n and src[m] == src[j]:
            m += 1
            j += 1

        # the last 5 bytes must stay literals
        if j > n - _LAST_LITERALS:
            j = n - _LAST_LITERALS
            if j <= i:
                break

        match_len = j - i
        lit_len = i - anchor
        offset = i - candidate

        # ---- token + literal length ----
        token_pos = len(out)
        if lit_len >= 15:
            out.append(0xF0)
            rest = lit_len - 15
            while rest >= 255:
                out.append(255)
                rest -= 255
            out.append(rest)
        else:
            out.append((lit_len << 4) & 0xF0)
        out += src[anchor:i]

        # ---- offset ----
        out.append(offset & 0xFF)
        out.append((offset >> 8) & 0xFF)

        # ---- match length ----
        ml = match_len - MIN_MATCH
        if ml >= 15:
            out[token_pos] |= 0x0F
            ml -= 15
            while ml >= 255:
                out.append(255)
                ml -= 255
            out.append(ml)
        else:
            out[token_pos] |= ml

        anchor = j
        i = j

    # ---- trailing literals ----
    lit_len = n - anchor
    if lit_len >= 15:
        out.append(0xF0)
        rest = lit_len - 15
        while rest >= 255:
            out.append(255)
            rest -= 255
        out.append(rest)
    else:
        out.append((lit_len << 4) & 0xF0)
    out += src[anchor:n]

    return bytes(out)
