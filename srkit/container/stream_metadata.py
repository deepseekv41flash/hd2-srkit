"""Stream-size policy for package parsing and patch editing.

A toc entry declares three blob sizes.  Real HD2 archives lie about the stream
size in two systematic ways, and a naive reader either slices garbage or raises
on a perfectly good package:

* ``Unit`` / ``Bones`` resources declare a non-zero stream size but have **no**
  stream payload at all — the documented size is meaningless;
* ``Texture`` resources put their streamed bytes in the first mip's
  ``BytesLeft`` field (toc body ``+0x10``), and some builds write the poison
  sentinel ``0xDFA64C92`` into the size field instead of the real length.

Ported from ``eigeen/hd2-mod-crates``
(``crates/hd2-archive-format/src/stream_metadata.rs``, MIT OR Apache-2.0) so
that packages written by that tool and by us agree byte for byte.

The policy is deliberately *reported*, never silent: every correction comes
back as a :class:`StreamMetadataRepair` so a caller can log it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..errors import FormatError
from ..util.ids import TYPE_BONES, TYPE_TEXTURE, TYPE_UNIT

__all__ = [
    "POISON_STREAM_SIZE",
    "StreamDeclaration",
    "StreamMetadataRepair",
    "ResolvedStreamSize",
    "resolve_stream_size",
]

#: size sentinel some builds write when the real length lives elsewhere
POISON_STREAM_SIZE = 0xDFA64C92


@dataclass
class StreamDeclaration:
    """What one toc entry claims about its stream blob."""

    file_id: int
    type_id: int
    offset: int
    size: int
    toc_data: bytes


@dataclass
class StreamMetadataRepair:
    """One entry whose declared stream size had to be recomputed."""

    file_id: int
    type_id: int
    original_size: int
    repaired_size: int
    poison_value_detected: bool


@dataclass
class ResolvedStreamSize:
    size: int
    repair: StreamMetadataRepair | None = None


def resolve_stream_size(
    declaration: StreamDeclaration,
    stream_len: int | None = None,
) -> ResolvedStreamSize:
    """Return the size the stream blob really has.

    ``stream_len=None`` means the caller only has a toc (no companion file);
    that is *not* the same as an empty stream and never becomes one — unless
    the type is known to carry no stream payload at all.
    """
    if not _declaration_mismatches(declaration, stream_len):
        return ResolvedStreamSize(size=declaration.size, repair=None)

    size = _calculate_actual_size(declaration, stream_len)
    if stream_len is not None and not _interval_in_bounds(declaration.offset, size, stream_len):
        raise FormatError(
            f"resource type=0x{declaration.type_id:016X}, file=0x{declaration.file_id:016X} "
            f"has a repaired stream size {size} that still runs past the "
            f"{stream_len} byte companion; offset={declaration.offset}"
        )
    return ResolvedStreamSize(
        size=size,
        repair=StreamMetadataRepair(
            file_id=declaration.file_id,
            type_id=declaration.type_id,
            original_size=declaration.size,
            repaired_size=size,
            poison_value_detected=declaration.size == POISON_STREAM_SIZE,
        ),
    )


def _declaration_mismatches(declaration: StreamDeclaration, stream_len: int | None) -> bool:
    if declaration.type_id in (TYPE_UNIT, TYPE_BONES):
        # these types never carry a stream blob, whatever the table says
        return declaration.size != 0
    if stream_len is None:
        return declaration.size == POISON_STREAM_SIZE
    return not _interval_in_bounds(declaration.offset, declaration.size, stream_len)


def _calculate_actual_size(declaration: StreamDeclaration, stream_len: int | None) -> int:
    if declaration.type_id in (TYPE_UNIT, TYPE_BONES) or stream_len == 0:
        return 0
    if declaration.type_id == TYPE_TEXTURE:
        if len(declaration.toc_data) < 0x14:
            raise FormatError(
                f"texture file=0x{declaration.file_id:016X}: toc body is "
                f"{len(declaration.toc_data)} bytes, need 20 to read the first "
                f"mip's BytesLeft"
            )
        return struct.unpack_from("<I", declaration.toc_data, 0x10)[0]
    raise FormatError(
        f"resource type=0x{declaration.type_id:016X}, file=0x{declaration.file_id:016X} "
        f"has mismatched stream metadata (offset={declaration.offset}, "
        f"size={declaration.size}, streamLength={stream_len}); "
        f"this type has no exact stream-size rule"
    )


def _interval_in_bounds(offset: int, size: int, stream_len: int) -> bool:
    return size == 0 or offset + size <= stream_len
