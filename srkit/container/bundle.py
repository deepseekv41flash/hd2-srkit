"""Stingray bundle sets: ``bundles.nxa`` index plus ``bundles.NN.nxa`` payloads.

``bundles.nxa`` is itself a DSAR file whose single resource is the index.  The
index (as decoded by ``utils/slim.py`` of the add-on) holds::

    0x00  12 bytes unknown
    0x0C  u32 number of bundle files
    0x10  u32 number of packages
    0x14  u32 unknown
    0x18  package descriptors, 24 bytes each
            +0x00 u64 size of the rebuilt package file
            +0x08 u32 name offset (into this index blob)
            +0x0C u32 item count
            +0x10 u32 items offset (into this index blob)
            +0x14 u32 unknown
    ...   NUL terminated package names
    ...   items, 16 bytes each, grouped per package
            +0x00 u64 offset inside the rebuilt package file
            +0x08 u32 offset inside the bundle's uncompressed content
            +0x0C 3 bytes unknown
            +0x0F u8  bundle file index

An item ("fragment") covers ``[original_archive_offset, next item's offset)`` of
the package file, so the items of one package must be sorted by their archive
offset.  The package file is rebuilt by copying each fragment out of its bundle.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

from ..errors import FormatError
from .dsar import DsarFile

__all__ = [
    "BundleItem", "BundlePackage", "BundleIndex", "BundleSet",
    "INDEX_HEADER_SIZE", "DESCRIPTOR_SIZE", "ITEM_SIZE",
    "BUNDLE_DATABASE_RECORD",
]

INDEX_HEADER_SIZE = 0x18
DESCRIPTOR_SIZE = 0x18
ITEM_SIZE = 0x10
BUNDLE_DATABASE_RECORD = 0x33
BUNDLE_DATABASE_NAME_TERMINATORS = (0x00, 0x17)


@dataclass
class BundleItem:
    original_archive_offset: int = 0
    uncompressed_bundle_offset: int = 0
    bundle_index: int = 0
    unknown: bytes = b"\x00\x00\x00"

    def serialize(self, stream) -> None:
        stream.uint64(self.original_archive_offset)
        stream.uint32(self.uncompressed_bundle_offset)
        stream.bytes(self.unknown, 3)
        stream.uint8(self.bundle_index)


@dataclass
class BundlePackage:
    name: str = ""
    size: int = 0
    items: list[BundleItem] = field(default_factory=list)
    unknown: int = 0

    def fragments(self) -> list[tuple[int, int]]:
        """``(archive_offset, fragment_size)`` for every item, in order."""
        ordered = sorted(self.items, key=lambda item: item.original_archive_offset)
        result = []
        for index, item in enumerate(ordered):
            if index + 1 < len(ordered):
                size = ordered[index + 1].original_archive_offset - item.original_archive_offset
            else:
                size = self.size - item.original_archive_offset
            result.append((item.original_archive_offset, size))
        return result


class BundleIndex:
    """Parsed ``bundles.nxa`` payload."""

    def __init__(self):
        self.header_prefix = bytearray(0x0C)
        self.num_bundles = 0
        self.num_packages = 0
        self.unknown_14 = 0
        self.packages: list[BundlePackage] = []

    @classmethod
    def parse(cls, data: bytes) -> "BundleIndex":
        if len(data) < INDEX_HEADER_SIZE:
            raise FormatError("bundles.nxa: index too small")
        index = cls()
        index.header_prefix = bytearray(data[:0x0C])
        index.num_bundles = struct.unpack_from("<I", data, 0x0C)[0]
        index.num_packages = struct.unpack_from("<I", data, 0x10)[0]
        index.unknown_14 = struct.unpack_from("<I", data, 0x14)[0]

        for i in range(index.num_packages):
            base = INDEX_HEADER_SIZE + i * DESCRIPTOR_SIZE
            if base + DESCRIPTOR_SIZE > len(data):
                raise FormatError(f"bundles.nxa: truncated at package {i}")
            size = int.from_bytes(data[base:base + 8], "little")
            name_offset, item_count, items_offset, unknown = struct.unpack_from(
                "<IIII", data, base + 8
            )
            name = _read_cstring(data, name_offset)
            package = BundlePackage(name=name, size=size, unknown=unknown)
            for j in range(item_count):
                off = items_offset + j * ITEM_SIZE
                if off + ITEM_SIZE > len(data):
                    raise FormatError(f"bundles.nxa: truncated item {j} of {name}")
                archive_offset = int.from_bytes(data[off:off + 8], "little")
                bundle_offset = struct.unpack_from("<I", data, off + 8)[0]
                package.items.append(
                    BundleItem(
                        original_archive_offset=archive_offset,
                        uncompressed_bundle_offset=bundle_offset,
                        bundle_index=data[off + 0x0F],
                        unknown=bytes(data[off + 0x0C:off + 0x0F]),
                    )
                )
            index.packages.append(package)
        return index

    def package(self, name: str) -> BundlePackage | None:
        for item in self.packages:
            if item.name == name:
                return item
        return None

    def to_bytes(self) -> bytes:
        descriptors_end = INDEX_HEADER_SIZE + len(self.packages) * DESCRIPTOR_SIZE

        names = bytearray()
        name_offsets: list[int] = []
        for package in self.packages:
            name_offsets.append(descriptors_end + len(names))
            names += package.name.encode("utf-8") + b"\x00"

        items_start = descriptors_end + len(names)
        items = bytearray()
        item_offsets: list[int] = []
        item_counts: list[int] = []
        for package in self.packages:
            item_offsets.append(items_start + len(items))
            item_counts.append(len(package.items))
            for item in package.items:
                entry = bytearray(ITEM_SIZE)
                struct.pack_into("<Q", entry, 0, item.original_archive_offset)
                struct.pack_into("<I", entry, 8, item.uncompressed_bundle_offset)
                entry[0x0C:0x0F] = item.unknown[:3].ljust(3, b"\x00")
                entry[0x0F] = item.bundle_index & 0xFF
                items += entry

        out = bytearray(descriptors_end + len(names) + len(items))
        out[0:0x0C] = self.header_prefix[:0x0C].ljust(0x0C, b"\x00")
        struct.pack_into("<III", out, 0x0C, self.num_bundles, len(self.packages), self.unknown_14)
        for i, package in enumerate(self.packages):
            base = INDEX_HEADER_SIZE + i * DESCRIPTOR_SIZE
            struct.pack_into("<Q", out, base, package.size)
            struct.pack_into(
                "<IIII", out, base + 8, name_offsets[i], item_counts[i], item_offsets[i],
                package.unknown,
            )
        out[descriptors_end:descriptors_end + len(names)] = names
        out[items_start:] = items
        return bytes(out)


def _read_cstring(data: bytes, offset: int, limit: int = 0x400) -> str:
    end = offset
    while end < len(data) and end - offset < limit and data[end] != 0:
        end += 1
    return data[offset:end].decode("utf-8", "replace")


class BundleSet:
    """Reader (and fragment level writer) for a ``bundles*.nxa`` set."""

    def __init__(self, root: str | os.PathLike, index_name: str = "bundles.nxa"):
        self.root = os.fspath(root)
        self.index_name = index_name
        self.index_path = os.path.join(self.root, index_name)
        self.index: BundleIndex | None = None
        self._bundle_cache: dict[int, DsarFile] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def is_bundle_set(cls, root: str | os.PathLike) -> bool:
        return os.path.isfile(os.path.join(os.fspath(root), "bundles.nxa"))

    def load_index(self) -> BundleIndex:
        if self.index is None:
            with open(self.index_path, "rb") as handle:
                data = handle.read()
            if DsarFile.is_dsar(data):
                payload = DsarFile.parse(data, self.index_path).read_resource(0)
            else:
                payload = data
            self.index = BundleIndex.parse(payload)
        return self.index

    def bundle_path(self, index: int) -> str:
        return os.path.join(self.root, f"bundles.{index:02d}.nxa")

    def bundle(self, index: int) -> DsarFile:
        if index not in self._bundle_cache:
            path = self.bundle_path(index)
            if not os.path.isfile(path):
                raise FormatError(f"bundle file missing: {path}")
            self._bundle_cache[index] = DsarFile.from_file(path)
        return self._bundle_cache[index]

    # ------------------------------------------------------------------ #
    def package_names(self) -> list[str]:
        return [package.name for package in self.load_index().packages]

    def read_package_range(self, name: str, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset`` of the rebuilt package file.

        Only the bundle fragments covering that range are touched, so pulling a
        single resource out of a multi hundred megabyte package stays cheap.
        """
        if size <= 0:
            return b""
        index = self.load_index()
        package = index.package(name)
        if package is None:
            raise FormatError(f"package '{name}' is not listed in {self.index_name}")

        out = bytearray()
        end = offset + size
        ordered = sorted(package.items, key=lambda item: item.original_archive_offset)
        for position, item in enumerate(ordered):
            start = item.original_archive_offset
            if position + 1 < len(ordered):
                frag_end = ordered[position + 1].original_archive_offset
            else:
                frag_end = package.size
            if frag_end <= offset or start >= end:
                continue
            take_start = max(offset, start)
            take_end = min(end, frag_end)
            bundle_offset = item.uncompressed_bundle_offset + (take_start - start)
            out += self.bundle(item.bundle_index).read_stream_range(
                bundle_offset, take_end - take_start
            )
        if len(out) != size:
            raise FormatError(
                f"{name}: wanted {size} bytes at {offset}, assembled {len(out)}"
            )
        return bytes(out)

    def read_fragment(self, item: BundleItem, size: int) -> bytes:
        bundle = self.bundle(item.bundle_index)
        if size <= 0:
            return b""
        # a fragment may span several consecutive resources
        return bundle.read_resource_span(item.uncompressed_bundle_offset, size)

    def extract_package(self, name: str) -> bytes:
        """Rebuild one package file (toc / gpu_resources / stream) byte for byte."""
        index = self.load_index()
        package = index.package(name)
        if package is None:
            raise FormatError(f"package '{name}' is not listed in {self.index_name}")
        data = bytearray(package.size)
        for item, (archive_offset, size) in zip(
            sorted(package.items, key=lambda i: i.original_archive_offset),
            package.fragments(),
        ):
            if size <= 0:
                continue
            chunk = self.read_fragment(item, size)
            data[archive_offset:archive_offset + len(chunk)] = chunk
        return bytes(data)

    def has_package(self, name: str) -> bool:
        return self.load_index().package(name) is not None

    # ------------------------------------------------------------------ #
    @staticmethod
    def build(
        packages: dict[str, bytes],
        chunk_size: int = 0x10000,
        max_bundle_size: int = 0x40000000,
        index_header_prefix: bytes | None = None,
        bundle_header_template: bytes | None = None,
        index_header_template: bytes | None = None,
        compress: bool = True,
    ) -> tuple[bytes, list[bytes], BundleIndex]:
        """Pack package blobs into a bundle set.

        Returns ``(index_payload, bundle_payloads, index)`` where the payloads are
        the *reconstructed* (uncompressed) contents.  Use :func:`write_bundle_set`
        to get the final DSAR files.
        """
        index = BundleIndex()
        if index_header_prefix:
            index.header_prefix = bytearray(index_header_prefix[:0x0C])

        bundle_contents: list[bytearray] = []
        bundle_resources: list[list[bytes]] = []

        def new_bundle() -> int:
            bundle_contents.append(bytearray())
            bundle_resources.append([])
            return len(bundle_contents) - 1

        current = new_bundle()
        for name, blob in packages.items():
            package = BundlePackage(name=name, size=len(blob))
            if not blob:
                # degenerate: an empty package still needs one fragment so that
                # the index keeps the name
                package.items.append(
                    BundleItem(0, len(bundle_contents[current]), current)
                )
                index.packages.append(package)
                continue

            offset = 0
            while offset < len(blob):
                room = max_bundle_size - len(bundle_contents[current])
                if room <= 0:
                    current = new_bundle()
                    room = max_bundle_size
                take = min(room, len(blob) - offset)
                fragment = blob[offset:offset + take]
                package.items.append(
                    BundleItem(
                        original_archive_offset=offset,
                        uncompressed_bundle_offset=len(bundle_contents[current]),
                        bundle_index=current,
                    )
                )
                bundle_contents[current] += fragment
                bundle_resources[current].append(fragment)
                offset += take
                if offset < len(blob):
                    current = new_bundle()
            index.packages.append(package)

        index.num_bundles = len(bundle_contents)
        bundle_payloads = [bytes(content) for content in bundle_contents]
        return index.to_bytes(), bundle_payloads, bundle_resources, index


def write_bundle_set(
    out_dir: str | os.PathLike,
    packages: dict[str, bytes],
    chunk_size: int = 0x10000,
    max_bundle_size: int = 0x40000000,
    bundle_header_template: bytes | None = None,
    index_header_template: bytes | None = None,
    index_header_prefix: bytes | None = None,
    compress: bool = True,
) -> BundleIndex:
    """Write ``bundles.nxa`` + ``bundles.NN.nxa`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)

    # two phase: build fragments first so that bundle payloads are known, then
    # DSAR-wrap them.  The index only needs the uncompressed offsets, so it can
    # be produced in the same pass.
    _, _, bundle_resources, index = BundleSet.build(
        packages,
        chunk_size=chunk_size,
        max_bundle_size=max_bundle_size,
        index_header_prefix=index_header_prefix,
        compress=compress,
    )

    index_payload = index.to_bytes()
    index_file = DsarFile.build(
        [index_payload],
        chunk_size=chunk_size,
        header_template=index_header_template,
        compress=compress,
        # retail pads the index payload to a 64 KiB boundary
        payload_alignment=0x10000,
    )
    with open(os.path.join(out_dir, "bundles.nxa"), "wb") as handle:
        handle.write(index_file)

    for i, resources in enumerate(bundle_resources):
        data = DsarFile.build(
            resources,
            chunk_size=chunk_size,
            header_template=bundle_header_template,
            compress=compress,
        )
        with open(os.path.join(out_dir, f"bundles.{i:02d}.nxa"), "wb") as handle:
            handle.write(data)

    return index


def read_bundle_database(path: str | os.PathLike) -> list[str]:
    """Read package names from ``bundle_database.data`` (HD2 retail layout)."""
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 0x10:
        return []
    count = struct.unpack_from("<I", data, 4)[0]
    names: list[str] = []
    for i in range(count):
        offset = 0x10 + i * BUNDLE_DATABASE_RECORD
        if offset + BUNDLE_DATABASE_RECORD > len(data):
            break
        record = data[offset:offset + BUNDLE_DATABASE_RECORD]
        cut = len(record)
        for terminator in BUNDLE_DATABASE_NAME_TERMINATORS:
            position = record.find(bytes([terminator]))
            if position != -1:
                cut = min(cut, position)
        names.append(record[:cut].decode("utf-8", "replace"))
    return names


def write_bundle_database(names: list[str], template: bytes | None = None) -> bytes:
    """Rebuild ``bundle_database.data`` from a package name list."""
    header = bytearray(0x10)
    if template and len(template) >= 0x10:
        header[:] = template[:0x10]
    struct.pack_into("<I", header, 4, len(names))
    out = bytearray(header)
    for name in names:
        record = bytearray(BUNDLE_DATABASE_RECORD)
        encoded = name.encode("utf-8")[:BUNDLE_DATABASE_RECORD - 1]
        record[0:len(encoded)] = encoded
        out += record
    return bytes(out)
