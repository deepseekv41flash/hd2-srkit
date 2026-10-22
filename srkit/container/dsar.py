"""``DSAR`` chunked container (Stingray bundles and slim packages).

Layout reverse engineered from the add-on (``utils/slim.py``)::

    0x00  u32  magic                      b"DSAR" == 0x52415344
    0x04  u32  version / unknown
    0x08  u32  chunk count
    0x0C  u32  unknown
    0x10  u64  unknown
    0x18  u64  unknown
    0x20  chunk table, 0x20 bytes per chunk
            u64 uncompressed_offset   (offset inside the concatenation of all chunks)
            u64 compressed_offset     (absolute offset of the payload in this file)
            u32 uncompressed_size
            u32 compressed_size
            u8  compression_type      0 = stored, 3 = LZ4 block
            u8  chunk_type            0x02 = START, 0x04 = CONTINUE, 0x01 = unknown
            6 bytes padding
    ...   chunk payloads

Chunks are independent: the engine therefore needs no global header beyond the
table.  ``chunk_type`` marks resource boundaries, which is what
``get_resource_from_bundle`` in the add-on relies on when a logical resource is
split over several chunks.
"""

from __future__ import annotations

import bisect
import os
import struct
from dataclasses import dataclass, field

from ..errors import FormatError
from ..util.binary import Stream, align_up
from ..util.ids import DSAR_MAGIC
from ..util.lz4block import compress as lz4_compress, decompress as lz4_decompress

__all__ = [
    "DsarFile", "DsarChunk",
    "COMPRESSION_NONE", "COMPRESSION_LZ4",
    "CHUNK_UNK", "CHUNK_START", "CHUNK_CONTINUE",
]

COMPRESSION_NONE = 0x00
COMPRESSION_LZ4 = 0x03

CHUNK_UNK = 0x01
CHUNK_START = 0x02
CHUNK_CONTINUE = 0x04

HEADER_SIZE = 0x20
CHUNK_ENTRY_SIZE = 0x20

#: header fields as observed in retail bundles (see docs/stingray-format.md §1)
DEFAULT_FORMAT_VERSION = 0x00010003
DEFAULT_FORMAT_ID = 3046489749524332880
#: retail bundle payloads are 16 byte aligned, the index payload 64 KiB aligned
PAYLOAD_ALIGNMENT = 0x10
INDEX_PAYLOAD_ALIGNMENT = 0x10000
#: the dominant retail chunk size (256 KiB)
DEFAULT_CHUNK_SIZE = 0x40000


@dataclass
class DsarChunk:
    uncompressed_offset: int = 0
    compressed_offset: int = 0
    uncompressed_size: int = 0
    compressed_size: int = 0
    compression_type: int = COMPRESSION_NONE
    chunk_type: int = CHUNK_START
    unknown: bytes = b"\x00" * 6

    @property
    def starts_resource(self) -> bool:
        return bool(self.chunk_type & CHUNK_START)

    @property
    def continues(self) -> bool:
        return bool(self.chunk_type & CHUNK_CONTINUE)

    def describe(self) -> str:
        kind = {COMPRESSION_NONE: "stored", COMPRESSION_LZ4: "lz4"}.get(
            self.compression_type, f"type{self.compression_type}"
        )
        flags = []
        if self.starts_resource:
            flags.append("start")
        if self.continues:
            flags.append("continue")
        if self.chunk_type & CHUNK_UNK:
            flags.append("unk")
        ratio = (
            f"{self.compressed_size / self.uncompressed_size:.2f}x"
            if self.uncompressed_size
            else "-"
        )
        return (
            f"@{self.compressed_offset:#x} raw={self.uncompressed_size} "
            f"stored={self.compressed_size} ({ratio}) {kind} "
            f"[{'+'.join(flags) or 'none'}]"
        )


@dataclass
class DsarFile:
    """Chunked container: header + chunk table + payload."""

    magic: int = DSAR_MAGIC
    version: int = 0
    unknown_0c: int = 0
    unknown_10: int = 0
    unknown_18: int = 0
    chunks: list[DsarChunk] = field(default_factory=list)
    raw: bytes = b""
    source: str = ""
    _offset_index: dict[int, int] | None = None
    _handle: object | None = None

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_dsar(data: bytes) -> bool:
        return len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == DSAR_MAGIC

    @classmethod
    def parse(cls, data: bytes, source: str = "") -> "DsarFile":
        if not cls.is_dsar(data):
            magic = struct.unpack_from("<I", data, 0)[0] if len(data) >= 4 else -1
            raise FormatError(
                f"{source or 'data'}: not a DSAR container (magic 0x{magic & 0xFFFFFFFF:08X})"
            )
        if len(data) < HEADER_SIZE:
            raise FormatError(f"{source or 'data'}: truncated DSAR header")

        magic, version, num_chunks, unknown_0c, unknown_10, unknown_18 = struct.unpack_from(
            "<IIIIQQ", data, 0
        )
        table_end = HEADER_SIZE + num_chunks * CHUNK_ENTRY_SIZE
        if table_end > len(data):
            raise FormatError(
                f"{source or 'data'}: chunk table needs {table_end} bytes, file has {len(data)}"
            )

        chunks: list[DsarChunk] = []
        for index in range(num_chunks):
            off = HEADER_SIZE + index * CHUNK_ENTRY_SIZE
            (
                uncompressed_offset,
                compressed_offset,
                uncompressed_size,
                compressed_size,
                compression_type,
                chunk_type,
                padding,
            ) = struct.unpack_from("<QQIIBB6s", data, off)
            chunks.append(
                DsarChunk(
                    uncompressed_offset=uncompressed_offset,
                    compressed_offset=compressed_offset,
                    uncompressed_size=uncompressed_size,
                    compressed_size=compressed_size,
                    compression_type=compression_type,
                    chunk_type=chunk_type,
                    unknown=padding,
                )
            )

        return cls(
            magic=magic,
            version=version,
            unknown_0c=unknown_0c,
            unknown_10=unknown_10,
            unknown_18=unknown_18,
            chunks=chunks,
            raw=data,
            source=source,
        )

    @classmethod
    def from_file(cls, path, use_mmap: bool = True, mmap_threshold: int = 8 << 20) -> "DsarFile":
        """Read a DSAR file, memory mapping large ones.

        Retail bundles are 500 MB - 1 GB each; mapping them keeps the reader's
        memory use proportional to the chunks it actually decodes instead of the
        file size.
        """
        import mmap as _mmap

        size = os.path.getsize(path)
        if use_mmap and size >= mmap_threshold:
            handle = open(path, "rb")
            try:
                data = _mmap.mmap(handle.fileno(), 0, access=_mmap.ACCESS_READ)
            except (OSError, ValueError):
                handle.close()
                with open(path, "rb") as plain:
                    return cls.parse(plain.read(), source=str(path))
            instance = cls.parse(data, source=str(path))
            instance._handle = handle
            return instance
        with open(path, "rb") as handle:
            return cls.parse(handle.read(), source=str(path))

    def close(self) -> None:
        raw = self.raw
        if hasattr(raw, "close"):
            try:
                raw.close()
            except (BufferError, ValueError):
                pass
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __enter__(self) -> "DsarFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # payload access
    # ------------------------------------------------------------------ #
    @property
    def header_template(self) -> bytes:
        """The 0x20 header bytes, reusable when rewriting a file."""
        if self.raw[:HEADER_SIZE] and len(self.raw) >= HEADER_SIZE:
            return self.raw[:HEADER_SIZE]
        return struct.pack(
            "<IIIIQQ",
            self.magic,
            self.version,
            len(self.chunks),
            self.unknown_0c,
            self.unknown_10,
            self.unknown_18,
        )

    def chunk_payload(self, index: int) -> bytes:
        chunk = self.chunks[index]
        start = chunk.compressed_offset
        end = start + chunk.compressed_size
        if end > len(self.raw):
            raise FormatError(
                f"chunk {index} payload out of range ({end} > {len(self.raw)})"
            )
        payload = self.raw[start:end]
        if chunk.compression_type == COMPRESSION_LZ4:
            return lz4_decompress(payload, chunk.uncompressed_size)
        if chunk.compression_type == COMPRESSION_NONE:
            return payload
        raise FormatError(f"chunk {index}: unsupported compression {chunk.compression_type}")

    def decompress_all(self) -> bytes:
        return b"".join(self.chunk_payload(i) for i in range(len(self.chunks)))

    def _chunk_index_for_offset(self, uncompressed_offset: int) -> int:
        if self._offset_index is None:
            self._offset_index = {
                chunk.uncompressed_offset: index
                for index, chunk in enumerate(self.chunks)
            }
        try:
            return self._offset_index[uncompressed_offset]
        except KeyError:
            raise FormatError(
                f"no chunk starts at uncompressed offset {uncompressed_offset}"
            ) from None

    def read_stream_range(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes of the *uncompressed stream* starting at ``offset``.

        Unlike :meth:`read_resource` this does not care about resource
        boundaries, which is what reading a single entry blob out of a huge
        bundle needs.
        """
        if size <= 0:
            return b""
        out = bytearray()
        remaining = size
        cursor = offset
        while remaining > 0:
            index = self._chunk_containing(cursor)
            chunk = self.chunks[index]
            payload = self.chunk_payload(index)
            delta = cursor - chunk.uncompressed_offset
            if delta >= len(payload):
                raise FormatError(
                    f"chunk {index} does not cover offset {cursor} "
                    f"({len(payload)} bytes from {chunk.uncompressed_offset})"
                )
            take = min(remaining, len(payload) - delta)
            out += payload[delta:delta + take]
            cursor += take
            remaining -= take
        return bytes(out)

    def _chunk_containing(self, offset: int) -> int:
        """Index of the chunk whose uncompressed range covers ``offset``."""
        if self._offset_index is None:
            self._offset_index = {
                chunk.uncompressed_offset: index
                for index, chunk in enumerate(self.chunks)
            }
        starts = sorted(self._offset_index)
        position = bisect.bisect_right(starts, offset) - 1
        if position < 0:
            raise FormatError(f"offset {offset} is before the first chunk")
        index = self._offset_index[starts[position]]
        chunk = self.chunks[index]
        if offset >= chunk.uncompressed_offset + chunk.uncompressed_size:
            raise FormatError(f"offset {offset} falls into a gap between chunks")
        return index

    def read_resource(self, start_offset: int, size: int | None = None) -> bytes:
        """Mirror of ``get_resource_from_bundle``: concatenate chunks from
        ``start_offset`` until the next chunk that starts a resource."""
        first = self._chunk_index_for_offset(start_offset)
        index = first
        parts: list[bytes] = []
        produced = 0
        while index < len(self.chunks):
            chunk = self.chunks[index]
            if index != first and chunk.starts_resource:
                break
            payload = self.chunk_payload(index)
            parts.append(payload)
            produced += len(payload)
            index += 1
            if size is not None and produced >= size:
                break
        data = b"".join(parts)
        if size is not None:
            if len(data) < size:
                raise FormatError(
                    f"resource at {start_offset} truncated: got {len(data)} of {size} bytes"
                )
            return data[:size]
        return data

    def read_resource_span(self, start_offset: int, size: int) -> bytes:
        """Read ``size`` bytes starting at ``start_offset``.

        A bundle fragment in the package index may cover *several* consecutive
        resources (the reference implementation accumulates them in
        ``get_resources_from_bundle``); this is the reader that turns a fragment
        back into bytes.
        """
        if size <= 0:
            return b""
        parts: list[bytes] = []
        produced = 0
        cursor = start_offset
        while produced < size:
            chunk = self.read_resource(cursor)
            if not chunk:
                raise FormatError(f"empty resource while reading {size} bytes at {cursor}")
            parts.append(chunk)
            produced += len(chunk)
            cursor += len(chunk)
        return b"".join(parts)[:size]

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    @staticmethod
    def build(
        resources: list[bytes],
        chunk_size: int = 0x40000,
        header_template: bytes | None = None,
        compress: bool = True,
        force_store_threshold: float = 1.0,
        payload_alignment: int = 16,
    ) -> bytes:
        """Create a DSAR file holding ``resources`` in order.

        Chunking and flags follow the retail layout (verified on 166k chunks of
        Helldivers 2 data):

        * a resource is split into ``chunk_size`` pieces, 256 KiB by default;
        * the first piece is flagged ``START`` (plus ``CONTINUE`` when more
          pieces follow), every following piece is flagged ``CONTINUE``;
        * payloads start right after the chunk table and are padded to
          ``payload_alignment`` (16 bytes in retail bundles, 64 KiB for the
          ``bundles.nxa`` index).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        chunks: list[DsarChunk] = []
        payloads: list[bytes] = []
        uncompressed_offset = 0

        for resource in resources:
            if not resource:
                continue
            pieces = [
                resource[pos:pos + chunk_size]
                for pos in range(0, len(resource), chunk_size)
            ]
            for piece_index, piece in enumerate(pieces):
                # first chunk starts the resource; CONTINUE marks "more follows"
                chunk_type = CHUNK_START if piece_index == 0 else CHUNK_CONTINUE
                if piece_index == 0 and len(pieces) > 1:
                    chunk_type |= CHUNK_CONTINUE

                payload = piece
                compression_type = COMPRESSION_NONE
                if compress:
                    packed = lz4_compress(piece)
                    if len(packed) < len(piece) * force_store_threshold:
                        payload = packed
                        compression_type = COMPRESSION_LZ4

                chunks.append(
                    DsarChunk(
                        uncompressed_offset=uncompressed_offset,
                        compressed_offset=0,  # patched below
                        uncompressed_size=len(piece),
                        compressed_size=len(payload),
                        compression_type=compression_type,
                        chunk_type=chunk_type,
                    )
                )
                payloads.append(payload)
                uncompressed_offset += len(piece)

        table_end = HEADER_SIZE + len(chunks) * CHUNK_ENTRY_SIZE
        cursor = align_up(table_end, payload_alignment)
        end = cursor
        for chunk, payload in zip(chunks, payloads):
            chunk.compressed_offset = cursor
            end = cursor + len(payload)
            cursor = align_up(end, payload_alignment)

        version = DEFAULT_FORMAT_VERSION
        format_id = DEFAULT_FORMAT_ID
        if header_template and len(header_template) >= HEADER_SIZE:
            version = struct.unpack_from("<I", header_template, 0x04)[0] or version
            format_id = struct.unpack_from("<Q", header_template, 0x18)[0] or format_id

        out = bytearray(end)
        if header_template and len(header_template) >= HEADER_SIZE:
            out[0:HEADER_SIZE] = header_template[:HEADER_SIZE]
        struct.pack_into(
            "<IIIIQQ",
            out,
            0,
            DSAR_MAGIC,
            version,
            len(chunks),
            table_end,
            sum(chunk.uncompressed_size for chunk in chunks),
            format_id,
        )

        for index, chunk in enumerate(chunks):
            off = HEADER_SIZE + index * CHUNK_ENTRY_SIZE
            struct.pack_into(
                "<QQIIBB6s",
                out,
                off,
                chunk.uncompressed_offset,
                chunk.compressed_offset,
                chunk.uncompressed_size,
                chunk.compressed_size,
                chunk.compression_type,
                chunk.chunk_type,
                chunk.unknown[:6].ljust(6, b"\x00"),
            )
        for chunk, payload in zip(chunks, payloads):
            out[chunk.compressed_offset:chunk.compressed_offset + len(payload)] = payload

        return bytes(out)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        stored = sum(c.compressed_size for c in self.chunks)
        raw = sum(c.uncompressed_size for c in self.chunks)
        resources = sum(1 for c in self.chunks if c.starts_resource)
        return (
            f"DSAR: {len(self.chunks)} chunk(s), {resources} resource start(s), "
            f"{raw} raw / {stored} stored bytes"
        )
