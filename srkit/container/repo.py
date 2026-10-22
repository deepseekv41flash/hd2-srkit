"""Unified access to the three container layouts Stingray ships in.

======================  =====================================================
layout                  files
======================  =====================================================
``loose``               ``<name>`` + ``<name>.gpu_resources`` + ``<name>.stream``
                        (plain bytes or DSAR wrapped, detected per file)
``bundled``             ``bundles.nxa`` + ``bundles.NN.nxa`` (+ optional
                        ``bundle_database.data`` with the package name list)
======================  =====================================================

The repository never decompresses gpu data unless a resource codec asks for it,
which matters because those buffers are multi gigabyte in retail data.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from ..errors import FormatError
from ..util.ids import DSAR_MAGIC, TOC_MAGIC
from .bundle import BundleSet, read_bundle_database
from .dsar import DsarFile
from .package import Package, TocEntry

__all__ = ["Repository", "PackageRef", "GPU_SUFFIX", "STREAM_SUFFIX"]

GPU_SUFFIX = ".gpu_resources"
STREAM_SUFFIX = ".stream"


@dataclass
class PackageRef:
    name: str
    kind: str            # "loose" | "dsar" | "bundled"
    path: str = ""


class Repository:
    """Read/write access to a set of packages."""

    def __init__(self, root: str, kind: str = "loose", explicit_name: str = ""):
        self.root = root
        self.kind = kind
        self.explicit_name = explicit_name
        self._bundles: BundleSet | None = None
        self._loose_cache: list[PackageRef] | None = None

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    @classmethod
    def open(cls, path: str | os.PathLike) -> "Repository":
        path = os.path.abspath(os.fspath(path))
        if not os.path.exists(path):
            raise FormatError(f"{path}: no such file or directory")
        if os.path.isdir(path):
            if BundleSet.is_bundle_set(path):
                return cls(path, "bundled")
            return cls(path, "loose")
        if not os.path.isfile(path):
            raise FormatError(f"{path}: not a regular file")

        directory = os.path.dirname(path)
        name = os.path.basename(path)
        if name.startswith("bundles.") and BundleSet.is_bundle_set(directory):
            return cls(directory, "bundled")
        with open(path, "rb") as handle:
            magic = handle.read(4)
        if len(magic) < 4:
            raise FormatError(f"{path}: file too small")
        value = struct.unpack("<I", magic)[0]
        if value in (TOC_MAGIC, DSAR_MAGIC):
            return cls(directory, "loose", explicit_name=name)
        raise FormatError(
            f"{path}: not a Stingray package (magic 0x{value:08X})"
        )

    # ------------------------------------------------------------------ #
    @property
    def bundles(self) -> BundleSet:
        if self._bundles is None:
            self._bundles = BundleSet(self.root)
        return self._bundles

    def package_refs(self) -> list[PackageRef]:
        if self.kind == "bundled":
            # the index also lists the companion streams (<name>.gpu_resources,
            # <name>.stream); only the toc entries are packages
            return [
                PackageRef(name=name, kind="bundled")
                for name in self.bundles.package_names()
                if not name.endswith(GPU_SUFFIX) and not name.endswith(STREAM_SUFFIX)
            ]
        if self._loose_cache is None:
            refs: list[PackageRef] = []
            if self.explicit_name:
                path = os.path.join(self.root, self.explicit_name)
                if self._is_package_file(path):
                    refs.append(PackageRef(self.explicit_name, "loose", path))
            else:
                for entry in sorted(os.listdir(self.root)):
                    if entry.endswith(GPU_SUFFIX) or entry.endswith(STREAM_SUFFIX):
                        continue
                    if entry.startswith("bundles.") or entry in ("bundle_database.data",):
                        continue
                    path = os.path.join(self.root, entry)
                    if self._is_package_file(path):
                        refs.append(PackageRef(entry, "loose", path))
            self._loose_cache = refs
        return self._loose_cache

    def declared_package_names(self) -> list[str]:
        """Names listed in ``bundle_database.data`` (may include absent content)."""
        if self.kind != "bundled":
            return [ref.name for ref in self.package_refs()]
        database = os.path.join(self.root, "bundle_database.data")
        if not os.path.isfile(database):
            return []
        return [
            name for name in read_bundle_database(database)
            if not name.endswith(GPU_SUFFIX) and not name.endswith(STREAM_SUFFIX)
        ]

    def package_names(self) -> list[str]:
        """Names that can actually be read from this repository."""
        names = [ref.name for ref in self.package_refs()]
        if self.kind == "bundled":
            # keep the database order for the entries that exist, then append
            # anything the index knows but the database does not list
            declared = [name for name in self.declared_package_names() if name in set(names)]
            names = declared + [name for name in names if name not in set(declared)]
        return names

    @staticmethod
    def _is_package_file(path: str) -> bool:
        try:
            if not os.path.isfile(path):
                return False
            with open(path, "rb") as handle:
                magic = handle.read(4)
        except OSError:
            return False
        if len(magic) < 4:
            return False
        return struct.unpack("<I", magic)[0] in (TOC_MAGIC, DSAR_MAGIC)

    # ------------------------------------------------------------------ #
    # payload access
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode_file(path: str) -> bytes:
        with open(path, "rb") as handle:
            data = handle.read()
        if not DsarFile.is_dsar(data):
            return data
        dsar = DsarFile.parse(data, path)
        return dsar.read_resource(0)

    def raw_streams(
        self, name: str, include_companions: bool = True
    ) -> tuple[bytes, bytes, bytes]:
        """Return the decompressed ``(toc, gpu, stream)`` payloads.

        ``include_companions=False`` skips the (potentially huge) gpu/stream
        blobs, which is what metadata listing needs.
        """
        if self.kind == "bundled":
            payload = self.bundles.extract_package(name)
            if not include_companions:
                return payload, b"", b""
            gpu = (
                self.bundles.extract_package(name + GPU_SUFFIX)
                if self.bundles.has_package(name + GPU_SUFFIX)
                else b""
            )
            stream = (
                self.bundles.extract_package(name + STREAM_SUFFIX)
                if self.bundles.has_package(name + STREAM_SUFFIX)
                else b""
            )
            return payload, gpu, stream

        path = os.path.join(self.root, name)
        if not os.path.isfile(path):
            raise FormatError(f"package '{name}' not found in {self.root}")
        toc = self._decode_file(path)
        if not include_companions:
            return toc, b"", b""
        gpu_path = path + GPU_SUFFIX
        stream_path = path + STREAM_SUFFIX
        gpu = self._decode_file(gpu_path) if os.path.isfile(gpu_path) else b""
        stream = self._decode_file(stream_path) if os.path.isfile(stream_path) else b""
        return toc, gpu, stream

    def companions(self, name: str) -> dict[str, bool]:
        """Which companion files exist for ``name`` (even when they are empty)."""
        if self.kind == "bundled":
            return {
                "gpu": self.bundles.has_package(name + GPU_SUFFIX),
                "stream": self.bundles.has_package(name + STREAM_SUFFIX),
            }
        path = os.path.join(self.root, name)
        return {
            "gpu": os.path.isfile(path + GPU_SUFFIX),
            "stream": os.path.isfile(path + STREAM_SUFFIX),
        }

    def load_package(self, name: str, load_blobs: bool = True) -> Package:
        toc, gpu, stream = self.raw_streams(name, include_companions=load_blobs)
        try:
            return Package.parse(toc, gpu, stream, name=name, load_blobs=load_blobs)
        except FormatError:
            # The DSAR resource flags may be used differently than assumed (the
            # reference reader stops at the next "start" chunk).  Retry with the
            # full concatenation of all chunks before giving up.
            if self.kind == "bundled":
                raise
            path = os.path.join(self.root, name)
            with open(path, "rb") as handle:
                data = handle.read()
            if not DsarFile.is_dsar(data):
                raise
            whole = DsarFile.parse(data, path).decompress_all()
            if whole == toc:
                raise
            return Package.parse(whole, gpu, stream, name=name, load_blobs=load_blobs)

    def load_entry(
        self, name: str, file_id: int, type_id: int | None = None
    ) -> TocEntry:
        """Load a single entry, reading only the bytes that entry occupies.

        Retail packages hold gigabytes of gpu data; this is the access path that
        makes "give me one texture/mesh" cheap.
        """
        file_id = int(file_id)
        package = self.load_package(name, load_blobs=False)
        candidates = [item for item in package.entries if item.file_id == file_id]
        if type_id is not None:
            candidates = [item for item in candidates if item.type_id == int(type_id)]
        if not candidates:
            raise FormatError(
                f"{name}: entry {file_id:016X} not found"
                + (f" with type {int(type_id):016X}" if type_id is not None else "")
            )
        entry = candidates[0]

        if self.kind == "bundled":
            # toc / gpu / stream live in *different* package files
            for attribute, offset_attribute, size_attribute, source in (
                ("toc_data", "toc_data_offset", "declared_toc_size", name),
                ("gpu_data", "gpu_resource_offset", "declared_gpu_size", name + GPU_SUFFIX),
                ("stream_data", "stream_offset", "declared_stream_size", name + STREAM_SUFFIX),
            ):
                size = getattr(entry, size_attribute)
                if not size:
                    continue
                if not self.bundles.has_package(source):
                    raise FormatError(
                        f"{name}: entry {file_id:016X} declares {size} bytes of "
                        f"{attribute} but companion package '{source}' is missing"
                    )
                offset = getattr(entry, offset_attribute)
                setattr(
                    entry, attribute,
                    bytearray(self.bundles.read_package_range(source, offset, size)),
                )
            return entry

        # loose layout: the whole file is small enough to read normally
        full = self.load_package(name, load_blobs=True)
        for item in full.entries:
            if item.file_id == file_id and (
                type_id is None or item.type_id == int(type_id)
            ):
                return item
        raise FormatError(f"{name}: entry {file_id:016X} vanished on reload")

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    def save_package(
        self,
        package: Package,
        out_dir: str | os.PathLike | None = None,
        compress: bool = False,
        chunk_size: int = 0x10000,
        name: str | None = None,
    ) -> list[str]:
        """Write a package back out in the loose layout.

        ``compress=False`` produces the plain (legacy) layout; ``True`` wraps
        every file in a DSAR container, which is what retail HD2 ships.
        """
        target = os.fspath(out_dir) if out_dir else self.root
        os.makedirs(target, exist_ok=True)
        package_name = name or package.name
        if not package_name:
            raise FormatError("package has no name; pass name=")

        toc, gpu, stream = package.to_bytes()
        written: list[str] = []

        def emit(path: str, payload: bytes) -> None:
            if not payload:
                return
            if compress:
                payload = DsarFile.build([payload], chunk_size=chunk_size)
            with open(path, "wb") as handle:
                handle.write(payload)
            written.append(path)

        emit(os.path.join(target, package_name), toc)
        emit(os.path.join(target, package_name + GPU_SUFFIX), gpu)
        emit(os.path.join(target, package_name + STREAM_SUFFIX), stream)
        return written

    def describe(self) -> str:
        if self.kind == "bundled":
            index = self.bundles.load_index()
            return (
                f"bundled repository {self.root}: {index.num_packages} package(s) "
                f"in {index.num_bundles} bundle file(s)"
            )
        return f"loose repository {self.root}: {len(self.package_refs())} package file(s)"
