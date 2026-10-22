"""The Stingray ``toc`` package (magic ``0xF0000011``).

A package is three byte streams that live next to each other::

    <name>                  toc header + entry table + per entry "toc" blob
    <name>.gpu_resources    per entry gpu blob   (64 byte aligned)
    <name>.stream           per entry stream blob (64 byte aligned)

Layout (cross checked against ``SearchToc``/``StreamToc``/``TocEntry`` in the
HD2SDK add-on)::

    0x00  u32 magic = 0xF0000011
    0x04  u32 type count
    0x08  u32 entry count
    0x0C  u32 unknown
    0x10  56 bytes unknown
    0x48  type table, 32 bytes per type
            u64 unknown, u64 type id, u64 entry count, u32 16, u32 64
    ...   entry table, 80 bytes per entry
            u64 file id, u64 type id,
            u64 toc offset, u64 stream offset, u64 gpu offset,
            u64 unknown1, u64 unknown2,
            u32 toc size, u32 stream size, u32 gpu size,
            u32 16, u32 64, u32 entry index
    ...   blobs (toc blobs are contiguous, gpu/stream are 64 byte aligned)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..errors import FormatError
from ..util.binary import Stream, align_up
from ..util.ids import TOC_MAGIC, type_name
from .stream_metadata import StreamDeclaration, StreamMetadataRepair, resolve_stream_size

__all__ = [
    "TocType", "TocEntry", "Package",
    "ALIGNMENT", "TOC_BLOB_ALIGNMENT", "MIN_ENTRY_BYTES",
]

ALIGNMENT = 64           # gpu/stream blobs are 64 byte aligned
TOC_BLOB_ALIGNMENT = 16  # toc blobs are 16 byte aligned in retail files
HEADER_SIZE = 0x48       # 72 bytes
TYPE_SIZE = 0x20         # 32 bytes
ENTRY_SIZE = 0x50        # 80 bytes
MIN_ENTRY_BYTES = 256    # the add-on pads small toc files to 256 bytes/entry


@dataclass
class TocType:
    type_id: int = 0
    num_files: int = 0
    unk1: int = 0
    unk2: int = 16
    unk3: int = 64

    def serialize(self, stream: Stream) -> None:
        stream.uint64(self.unk1)
        stream.uint64(self.type_id)
        stream.uint64(self.num_files)
        stream.uint32(self.unk2)
        stream.uint32(self.unk3)


@dataclass
class TocEntry:
    """One resource: three independent byte blobs."""

    file_id: int = 0
    type_id: int = 0
    toc_data: bytearray = field(default_factory=bytearray)
    gpu_data: bytearray = field(default_factory=bytearray)
    stream_data: bytearray = field(default_factory=bytearray)

    # header bookkeeping (preserved for byte faithful round trips)
    toc_data_offset: int = 0
    stream_offset: int = 0
    gpu_resource_offset: int = 0
    unknown1: int = 0
    unknown2: int = 0
    unknown3: int = 16
    unknown4: int = 64
    entry_index: int = 0

    # declared sizes as found in the entry table (0 = derive from the blob)
    declared_toc_size: int = 0
    declared_gpu_size: int = 0
    declared_stream_size: int = 0

    #: True when ``entry_index`` came from a parsed file (so 0 is kept as is)
    entry_index_known: bool = False

    # runtime helper (never serialized)
    source_package: str = ""

    @property
    def label(self) -> str:
        return f"{self.file_id:016X}"

    def set_data(self, toc=None, gpu=None, stream=None) -> None:
        if toc is not None:
            self.toc_data = bytearray(toc)
        if gpu is not None:
            self.gpu_data = bytearray(gpu)
        if stream is not None:
            self.stream_data = bytearray(stream)

    def sizes(self) -> tuple[int, int, int]:
        return len(self.toc_data), len(self.gpu_data), len(self.stream_data)

    def describe(self, names=None) -> str:
        label = names.label(self.file_id) if names else f"{self.file_id:016X}"
        kind = type_name(self.type_id)
        toc, gpu, stream = self.sizes()
        return f"{kind:<15} {label:<44} toc={toc:<9} gpu={gpu:<9} stream={stream}"


class Package:
    """A parsed ``<name>`` / ``.gpu_resources`` / ``.stream`` triple."""

    def __init__(self, name: str = ""):
        self.name = name
        self.magic = TOC_MAGIC
        self.unknown = 0
        self.unk4_data = bytearray(56)
        self.types: list[TocType] = []
        self.entries: list[TocEntry] = []
        #: declared stream sizes that had to be recomputed while parsing
        self.stream_repairs: list[StreamMetadataRepair] = []

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #
    def entry(self, file_id: int, type_id: int) -> TocEntry | None:
        file_id = int(file_id)
        type_id = int(type_id)
        for item in self.entries:
            if item.file_id == file_id and item.type_id == type_id:
                return item
        return None

    def entries_of_type(self, type_id: int) -> list[TocEntry]:
        type_id = int(type_id)
        return [item for item in self.entries if item.type_id == type_id]

    def add_entry(self, entry: TocEntry, replace: bool = False) -> TocEntry:
        existing = self.entry(entry.file_id, entry.type_id)
        if existing is not None:
            if not replace:
                raise FormatError(
                    f"package {self.name}: entry {entry.file_id:016X}/"
                    f"{type_name(entry.type_id)} already exists"
                )
            self.entries[self.entries.index(existing)] = entry
            return entry
        self.entries.append(entry)
        if not any(t.type_id == entry.type_id for t in self.types):
            self.types.append(TocType(type_id=entry.type_id))
        return entry

    def remove_entry(self, file_id: int, type_id: int) -> bool:
        item = self.entry(file_id, type_id)
        if item is None:
            return False
        self.entries.remove(item)
        return True

    def ordered_entries(self) -> list[TocEntry]:
        """Entries grouped by the type table order (what the writer emits)."""
        ordered: list[TocEntry] = []
        seen: set[int] = set()
        for toc_type in self.types:
            for item in self.entries:
                if item.type_id == toc_type.type_id:
                    ordered.append(item)
                    seen.add(id(item))
        for item in self.entries:  # safety net for unregistered types
            if id(item) not in seen:
                ordered.append(item)
        return ordered

    def stats(self) -> dict:
        per_type: dict[str, int] = {}
        for item in self.entries:
            key = type_name(item.type_id)
            per_type[key] = per_type.get(key, 0) + 1
        return {
            "name": self.name,
            "entries": len(self.entries),
            "types": per_type,
            "toc_bytes": sum(len(e.toc_data) for e in self.entries),
            "gpu_bytes": sum(len(e.gpu_data) for e in self.entries),
            "stream_bytes": sum(len(e.stream_data) for e in self.entries),
        }

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    @classmethod
    def parse(
        cls,
        toc_data: bytes | bytearray,
        gpu_data: bytes | bytearray | None = None,
        stream_data: bytes | bytearray | None = None,
        name: str = "",
        load_blobs: bool = True,
    ) -> "Package":
        """Parse a toc file (optionally with its gpu/stream companions).

        ``load_blobs=False`` parses the tables only, which is what listing needs:
        no multi gigabyte gpu buffer has to be touched.
        """
        if len(toc_data) < HEADER_SIZE:
            raise FormatError(
                f"{name or 'package'}: too small for a toc header ({len(toc_data)} bytes)"
            )
        magic, num_types, num_files, unknown = struct.unpack_from("<IIII", toc_data, 0)
        if magic != TOC_MAGIC:
            raise FormatError(
                f"{name or 'package'}: bad toc magic 0x{magic:08X} (expected 0x{TOC_MAGIC:08X})"
            )

        package = cls(name)
        package.magic = magic
        package.unknown = unknown
        package.unk4_data = bytearray(toc_data[0x10:0x10 + 56])

        cursor = HEADER_SIZE
        for _ in range(num_types):
            if cursor + TYPE_SIZE > len(toc_data):
                raise FormatError(f"{name}: truncated type table")
            unk1, type_id, type_files, unk2, unk3 = struct.unpack_from("<QQQII", toc_data, cursor)
            package.types.append(
                TocType(type_id=type_id, num_files=type_files, unk1=unk1, unk2=unk2, unk3=unk3)
            )
            cursor += TYPE_SIZE

        for _ in range(num_files):
            if cursor + ENTRY_SIZE > len(toc_data):
                raise FormatError(f"{name}: truncated entry table")
            (
                file_id,
                type_id,
                toc_offset,
                stream_offset,
                gpu_offset,
                unknown1,
                unknown2,
                toc_size,
                stream_size,
                gpu_size,
                unknown3,
                unknown4,
                entry_index,
            ) = struct.unpack_from("<QQQQQQQIIIIII", toc_data, cursor)
            package.entries.append(
                TocEntry(
                    file_id=file_id,
                    type_id=type_id,
                    toc_data_offset=toc_offset,
                    stream_offset=stream_offset,
                    gpu_resource_offset=gpu_offset,
                    unknown1=unknown1,
                    unknown2=unknown2,
                    unknown3=unknown3,
                    unknown4=unknown4,
                    entry_index=entry_index,
                    declared_toc_size=toc_size,
                    declared_gpu_size=gpu_size,
                    declared_stream_size=stream_size,
                    entry_index_known=True,
                    source_package=name,
                )
            )
            cursor += ENTRY_SIZE

        if load_blobs:
            _slice_blobs(toc_data, package.entries, "toc_data", "toc_data_offset",
                         "declared_toc_size", name, "toc")
            if gpu_data is not None:
                _slice_blobs(gpu_data, package.entries, "gpu_data", "gpu_resource_offset",
                             "declared_gpu_size", name, "gpu")
            if stream_data is not None:
                _slice_stream_blobs(stream_data, package.entries, package.stream_repairs, name)

        return package

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def to_bytes(self, pad_min_size: bool = True, stream_align: int = 0) -> tuple[bytes, bytes, bytes]:
        """Serialize back to ``(toc, gpu_resources, stream)``.

        ``stream_align`` defaults to 0 (retail appends stream blobs back to
        back); pass 64 to match the layout the community migrator writes.
        """
        ordered = self.ordered_entries()

        # type table: declared types first (file order), then any type that only
        # shows up in the entry list (possible after manual edits).
        types: list[TocType] = list(self.types)
        known = {t.type_id for t in types}
        for item in ordered:
            if item.type_id not in known:
                types.append(TocType(type_id=item.type_id))
                known.add(item.type_id)

        counts: dict[int, int] = {}
        for item in ordered:
            counts[item.type_id] = counts.get(item.type_id, 0) + 1
        for toc_type in types:
            toc_type.num_files = counts.get(toc_type.type_id, 0)

        toc = Stream.writer()
        gpu = Stream.writer()
        stream = Stream.writer()

        toc.uint32(self.magic)
        toc.uint32(len(types))
        toc.uint32(len(ordered))
        toc.uint32(self.unknown)
        toc.bytes(self.unk4_data, 56)
        for toc_type in types:
            toc_type.serialize(toc)

        table_start = toc.tell()
        toc.seek(table_start + len(ordered) * ENTRY_SIZE)  # reserve entry table

        # Real game packages start every toc blob on a 16 byte boundary (verified
        # against a retail patch archive).  When the original offsets are known we
        # keep them, which also reproduces the layout of files written without
        # alignment (e.g. by the Blender add-on).
        for item in ordered:
            hint = item.toc_data_offset
            if hint and hint >= toc.tell():
                toc.seek(hint)
            else:
                toc.seek(align_up(toc.tell(), TOC_BLOB_ALIGNMENT))
            item.toc_data_offset = toc.tell()
            toc.bytes(item.toc_data)

            hint = item.gpu_resource_offset
            target = hint if hint and hint >= gpu.tell() else align_up(gpu.tell(), ALIGNMENT)
            item.gpu_resource_offset = target
            if item.gpu_data:
                gpu.seek(target)
                gpu.bytes(item.gpu_data)

            # retail stream blobs are appended without extra alignment (verified
            # on a real package: offset 1398128 right after a 1398128 byte blob)
            hint = item.stream_offset
            if hint and hint >= stream.tell():
                target = hint
            elif stream_align:
                target = align_up(stream.tell(), stream_align)
            else:
                target = stream.tell()
            item.stream_offset = target
            if item.stream_data:
                stream.seek(target)
                stream.bytes(item.stream_data)

        # rewrite the entry table with the final offsets
        toc.seek(table_start)
        for index, item in enumerate(ordered):
            if not item.entry_index and not item.entry_index_known:
                item.entry_index = index + 1
            toc.uint64(item.file_id)
            toc.uint64(item.type_id)
            toc.uint64(item.toc_data_offset)
            toc.uint64(item.stream_offset)
            toc.uint64(item.gpu_resource_offset)
            toc.uint64(item.unknown1)
            toc.uint64(item.unknown2)
            toc.uint32(len(item.toc_data))
            toc.uint32(len(item.stream_data))
            toc.uint32(len(item.gpu_data))
            toc.uint32(item.unknown3)
            toc.uint32(item.unknown4)
            toc.uint32(item.entry_index)
        toc.seek(len(toc.Data))
        minimum = align_up(len(toc.Data), TOC_BLOB_ALIGNMENT)
        if pad_min_size:
            minimum = max(minimum, MIN_ENTRY_BYTES * max(1, len(ordered)))
        toc.seek(minimum)

        return toc.getvalue(), gpu.getvalue(), stream.getvalue()


def _slice_stream_blobs(
    container: bytes | bytearray,
    entries: list[TocEntry],
    repairs: list[StreamMetadataRepair],
    package_name: str,
) -> None:
    """Slice stream blobs, repairing the two stream sizes HD2 declares wrong.

    ``Unit``/``Bones`` entries must end up with an empty stream and textures
    store their real length in the first mip; see ``stream_metadata``.  Every
    correction is appended to ``repairs`` so the caller can report it.
    """
    for item in entries:
        declaration = StreamDeclaration(
            file_id=item.file_id,
            type_id=item.type_id,
            offset=item.stream_offset,
            size=item.declared_stream_size,
            toc_data=bytes(item.toc_data),
        )
        resolved = resolve_stream_size(declaration, len(container))
        if resolved.repair is not None:
            repairs.append(resolved.repair)
            item.declared_stream_size = resolved.size
        if resolved.size <= 0:
            continue
        end = item.stream_offset + resolved.size
        if end > len(container):
            raise FormatError(
                f"{package_name}: entry {item.file_id:016X} stream blob "
                f"[{item.stream_offset}:{end}] exceeds container size {len(container)}"
            )
        item.stream_data = bytearray(container[item.stream_offset:end])


def _slice_blobs(
    container: bytes | bytearray,
    entries: list[TocEntry],
    attribute: str,
    offset_attribute: str,
    size_attribute: str,
    package_name: str,
    what: str,
) -> None:
    for item in entries:
        offset = getattr(item, offset_attribute)
        size = getattr(item, size_attribute)
        if size <= 0:
            continue
        end = offset + size
        if end > len(container):
            raise FormatError(
                f"{package_name}: entry {item.file_id:016X} {what} blob "
                f"[{offset}:{end}] exceeds container size {len(container)}"
            )
        setattr(item, attribute, bytearray(container[offset:end]))
