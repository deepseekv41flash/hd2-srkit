"""Unpack a Stingray archive to a directory tree and pack it back.

Layout produced by :func:`unpack`::

    <out>/manifest.json
    <out>/<package>/package.json
    <out>/<package>/resources/<type>_<fileid>/resource.json
    <out>/<package>/resources/<type>_<fileid>/<payload files>

``pack`` rebuilds the three blobs of every entry through its codec and writes the
package again.  Entries that were not unpacked (filtered out) are copied verbatim
from the source repository, so a partial unpack still produces a complete, valid
package ("patch" workflow).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable

from .container.package import Package, TocEntry, TocType
from .container.repo import Repository
from .errors import CodecError, FormatError, SrkitError
from .resources import decode_entry, encode_entry, is_registered, get_codec
from .util.ids import NameTable, type_name, type_extension

__all__ = ["unpack", "pack", "verify", "MANIFEST_NAME", "PACKAGE_FILE", "RESOURCE_FILE"]

MANIFEST_NAME = "manifest.json"
PACKAGE_FILE = "package.json"
RESOURCE_FILE = "resource.json"
WORKSPACE_FORMAT = "srkit/workspace"
WORKSPACE_VERSION = 1

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    cleaned = _ILLEGAL.sub("_", name).strip(" .")
    return cleaned or "unnamed"


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _summary(blob: bytes) -> dict:
    return {"size": len(blob), "sha256": _sha(blob)[:16]}


# --------------------------------------------------------------------------- #
# unpack
# --------------------------------------------------------------------------- #
def unpack(
    source: str,
    out_dir: str,
    names: NameTable | None = None,
    packages: list[str] | None = None,
    types: list[int] | None = None,
    file_ids: list[int] | None = None,
    include_gpu: bool = True,
    include_stream: bool = True,
    raw: bool = False,
    decode_vertices: bool = False,
    overwrite: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> dict:
    """Extract resources from ``source`` into ``out_dir``."""
    names = names or NameTable()
    repo = Repository.open(source)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise SrkitError(
            f"{out_dir} already exists and is not empty (use overwrite=True / --force)"
        )
    os.makedirs(out_dir, exist_ok=True)

    selected = packages or repo.package_names()
    manifest = {
        "format": WORKSPACE_FORMAT,
        "version": WORKSPACE_VERSION,
        "source": {"path": os.path.abspath(source), "kind": repo.kind},
        "filters": {
            "packages": packages,
            "types": [str(item) for item in types] if types else None,
            "file_ids": [str(item) for item in file_ids] if file_ids else None,
            "gpu": include_gpu,
            "stream": include_stream,
            "raw": raw,
        },
        "packages": [],
    }

    totals = {"packages": 0, "entries": 0, "written": 0, "skipped": 0, "errors": 0,
              "payload_bytes": 0}
    for package_name in selected:
        try:
            package = repo.load_package(package_name, load_blobs=True)
        except (FormatError, SrkitError) as error:
            log(f"!! {package_name}: {error}")
            totals["errors"] += 1
            continue

        package_dir = os.path.join(out_dir, safe_name(package_name))
        resources_dir = os.path.join(package_dir, "resources")
        os.makedirs(resources_dir, exist_ok=True)

        package_record = _package_to_json(package)
        package_record["companions"] = repo.companions(package_name)
        written_entries = []
        for entry in package.entries:
            if types and entry.type_id not in types:
                totals["skipped"] += 1
                continue
            if file_ids and entry.file_id not in file_ids:
                totals["skipped"] += 1
                continue

            resource_dir_name = f"{type_name(entry.type_id)}_{entry.file_id:016X}"
            resource_dir = os.path.join(resources_dir, resource_dir_name)
            os.makedirs(resource_dir, exist_ok=True)

            record = _entry_to_json(entry, resource_dir_name)
            try:
                if raw:
                    decoded_ir, payloads = _raw_decode(entry)
                else:
                    decoded = decode_entry(
                        entry, names, {"decode_vertices": decode_vertices}
                    )
                    decoded_ir, payloads = decoded.ir, decoded.payloads
            except SrkitError as error:
                log(f"!! {package_name}/{entry.file_id:016X}: {error} (kept raw)")
                totals["errors"] += 1
                decoded_ir, payloads = _raw_decode(entry)
                record["decode_error"] = str(error)

            payload_records = []
            for payload in payloads:
                if payload.name == "gpu.bin" and not include_gpu:
                    continue
                if payload.name == "stream.bin" and not include_stream:
                    continue
                if payload.view and not decode_vertices:
                    continue
                with open(os.path.join(resource_dir, payload.name), "wb") as handle:
                    handle.write(payload.data)
                payload_records.append(
                    {
                        "name": payload.name,
                        "size": len(payload.data),
                        "editable": payload.editable,
                        "view": payload.view,
                        "note": payload.note,
                        "sha256": _sha(payload.data)[:16],
                    }
                )
                totals["payload_bytes"] += len(payload.data)

            decoded_ir["payloads"] = payload_records
            decoded_ir["source"] = {
                "package": package_name,
                "file_id": str(entry.file_id),
                "type_id": str(entry.type_id),
            }
            _write_json(os.path.join(resource_dir, RESOURCE_FILE), decoded_ir)
            record["codec"] = decoded_ir.get("format", "srkit/raw").split("/")[-1]
            record["read_only"] = bool(decoded_ir.get("read_only"))
            written_entries.append(record)
            totals["written"] += 1

        package_record["entries"] = written_entries
        _write_json(os.path.join(package_dir, PACKAGE_FILE), package_record)
        manifest["packages"].append(
            {
                "name": package_name,
                "directory": safe_name(package_name),
                "entries": len(package.entries),
                "unpacked": len(written_entries),
            }
        )
        totals["packages"] += 1
        totals["entries"] += len(package.entries)
        log(
            f"   {package_name}: {len(written_entries)}/{len(package.entries)} "
            f"resource(s) unpacked"
        )

    _write_json(os.path.join(out_dir, MANIFEST_NAME), manifest)
    return totals


def _raw_decode(entry: TocEntry):
    codec = get_codec(entry.type_id)
    if codec is None or codec.name != "raw":
        from .resources.raw import raw_codec

        codec = raw_codec()
    decoded = codec.decode(entry, NameTable())
    return decoded.ir, decoded.payloads


def _package_to_json(package: Package) -> dict:
    return {
        "format": "srkit/package",
        "name": package.name,
        "magic": package.magic,
        "unknown": package.unknown,
        "unk4_data_hex": bytes(package.unk4_data).hex(),
        "types": [
            {
                "type_id": str(item.type_id),
                "type_name": type_name(item.type_id),
                "num_files": item.num_files,
                "unk1": str(item.unk1),
                "unk2": item.unk2,
                "unk3": item.unk3,
            }
            for item in package.types
        ],
        "entries": [],
    }


def _entry_to_json(entry: TocEntry, resource_dir: str) -> dict:
    return {
        "file_id": str(entry.file_id),
        "type_id": str(entry.type_id),
        "type_name": type_name(entry.type_id),
        "resource_dir": f"resources/{resource_dir}",
        "toc_data_offset": entry.toc_data_offset,
        "stream_offset": entry.stream_offset,
        "gpu_resource_offset": entry.gpu_resource_offset,
        "unknown1": str(entry.unknown1),
        "unknown2": str(entry.unknown2),
        "unknown3": entry.unknown3,
        "unknown4": entry.unknown4,
        "entry_index": entry.entry_index,
        "declared_toc_size": entry.declared_toc_size,
        "declared_gpu_size": entry.declared_gpu_size,
        "declared_stream_size": entry.declared_stream_size,
        "blobs": {
            "toc": _summary(entry.toc_data),
            "gpu": _summary(entry.gpu_data),
            "stream": _summary(entry.stream_data),
        },
    }


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
def pack(
    workspace: str,
    out_dir: str,
    names: NameTable | None = None,
    source: str | None = None,
    compress: bool = False,
    chunk_size: int = 0x10000,
    packages: list[str] | None = None,
    output_name: str | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> dict:
    """Rebuild packages from an unpacked workspace.

    ``output_name`` renames the written package files, which is how a Helldivers 2
    mod archive name is produced (``--name 9ba626afa44a3aa3.patch_0``).
    """
    names = names or NameTable()
    manifest_path = os.path.join(workspace, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise SrkitError(f"{workspace} is not an srkit workspace (no {MANIFEST_NAME})")
    manifest = _read_json(manifest_path)

    source_path = source or (manifest.get("source") or {}).get("path")
    original_repo: Repository | None = None
    if source_path and os.path.exists(source_path):
        try:
            original_repo = Repository.open(source_path)
        except SrkitError:
            original_repo = None

    os.makedirs(out_dir, exist_ok=True)
    totals = {"packages": 0, "entries": 0, "reencoded": 0, "copied": 0, "errors": 0}

    for record in manifest.get("packages", []):
        if packages and record["name"] not in packages:
            continue
        package_dir = os.path.join(workspace, record["directory"])
        package_json = os.path.join(package_dir, PACKAGE_FILE)
        if not os.path.isfile(package_json):
            log(f"!! {record['name']}: missing {PACKAGE_FILE}, skipped")
            totals["errors"] += 1
            continue
        data = _read_json(package_json)

        package = Package(data.get("name") or record["name"])
        package.magic = int(data.get("magic", package.magic))
        package.unknown = int(data.get("unknown", 0))
        package.unk4_data = bytearray(bytes.fromhex(data.get("unk4_data_hex", "")))
        package.types = [
            TocType(
                type_id=int(item["type_id"]),
                num_files=int(item.get("num_files", 0)),
                unk1=int(item.get("unk1", 0)),
                unk2=int(item.get("unk2", 16)),
                unk3=int(item.get("unk3", 64)),
            )
            for item in data.get("types", [])
        ]

        for item in data.get("entries", []):
            entry = TocEntry(
                file_id=int(item["file_id"]),
                type_id=int(item["type_id"]),
                unknown1=int(item.get("unknown1", 0)),
                unknown2=int(item.get("unknown2", 0)),
                unknown3=int(item.get("unknown3", 16)),
                unknown4=int(item.get("unknown4", 64)),
                entry_index=int(item.get("entry_index", 0)),
                entry_index_known=True,
                declared_toc_size=int(item.get("declared_toc_size", 0)),
                declared_gpu_size=int(item.get("declared_gpu_size", 0)),
                declared_stream_size=int(item.get("declared_stream_size", 0)),
                # preserving the original offsets reproduces the retail blob
                # layout (including its 16 byte / larger paddings) exactly
                toc_data_offset=int(item.get("toc_data_offset", 0)),
                gpu_resource_offset=int(item.get("gpu_resource_offset", 0)),
                stream_offset=int(item.get("stream_offset", 0)),
                source_package=package.name,
            )
            resource_dir = item.get("resource_dir")
            resource_path = os.path.join(package_dir, resource_dir) if resource_dir else ""
            ir_path = os.path.join(resource_path, RESOURCE_FILE)

            if resource_dir and os.path.isfile(ir_path):
                try:
                    toc, gpu, stream = _encode_resource(
                        ir_path, resource_path, entry, names
                    )
                    entry.set_data(toc, gpu, stream)
                    totals["reencoded"] += 1
                except SrkitError as error:
                    log(f"!! {package.name}/{entry.file_id:016X}: {error}")
                    totals["errors"] += 1
                    _copy_from_source(entry, original_repo, package.name)
                    totals["copied"] += 1
            else:
                if not _copy_from_source(entry, original_repo, package.name):
                    raise SrkitError(
                        f"{package.name}/{entry.file_id:016X}: no resource directory "
                        f"and no source to copy from"
                    )
                totals["copied"] += 1
            package.entries.append(entry)

        written = _save(
            package=package, out_dir=out_dir, compress=compress,
            chunk_size=chunk_size, output_name=output_name,
            companions=data.get("companions") or {},
        )
        totals["packages"] += 1
        totals["entries"] += len(package.entries)
        log(f"   {package.name}: {len(package.entries)} resource(s) -> "
            f"{', '.join(os.path.basename(path) for path in written)}")

    return totals


def _save(
    package: Package,
    out_dir: str,
    compress: bool,
    chunk_size: int,
    output_name: str | None = None,
    companions: dict | None = None,
):
    from .container.repo import GPU_SUFFIX, STREAM_SUFFIX
    from .container.dsar import DsarFile

    toc, gpu, stream = package.to_bytes()
    written = []
    name = output_name or package.name
    companions = companions or {}

    def emit(path: str, payload: bytes, force: bool = False) -> None:
        if not payload and not force:
            return
        if compress and payload:
            payload = DsarFile.build([payload], chunk_size=chunk_size)
        with open(path, "wb") as handle:
            handle.write(payload)
        written.append(path)

    emit(os.path.join(out_dir, name), toc)
    # keep empty companion files when the source had them, so the output layout
    # matches the original archive exactly
    emit(os.path.join(out_dir, name + GPU_SUFFIX), gpu, force=bool(companions.get("gpu")))
    emit(os.path.join(out_dir, name + STREAM_SUFFIX), stream,
         force=bool(companions.get("stream")))
    return written


def _copy_from_source(entry: TocEntry, repo: Repository | None, package_name: str) -> bool:
    if repo is None:
        return False
    try:
        package = repo.load_package(package_name, load_blobs=True)
    except SrkitError:
        return False
    original = package.entry(entry.file_id, entry.type_id)
    if original is None:
        return False
    entry.set_data(original.toc_data, original.gpu_data, original.stream_data)
    entry.toc_data_offset = original.toc_data_offset
    entry.gpu_resource_offset = original.gpu_resource_offset
    entry.stream_offset = original.stream_offset
    entry.unknown1 = original.unknown1
    entry.unknown2 = original.unknown2
    entry.unknown3 = original.unknown3
    entry.unknown4 = original.unknown4
    entry.entry_index = original.entry_index
    return True


def _encode_resource(
    ir_path: str, resource_dir: str, entry: TocEntry, names: NameTable
) -> tuple[bytes, bytes, bytes]:
    ir = _read_json(ir_path)
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for record in ir.get("payloads", []):
        name = record.get("name")
        if not name or record.get("view"):
            continue
        path = os.path.join(resource_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                payloads[name] = handle.read()
        else:
            missing.append(name)
    if missing:
        raise CodecError(
            f"payload file(s) missing: {', '.join(missing)} "
            f"(restore them or unpack again)"
        )
    # encode_entry follows ir["format"], so a workspace unpacked with --raw is
    # re-packed with the raw codec even for types that have a dedicated codec
    return encode_entry(ir, payloads, entry, names)


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
@dataclass
class VerifyResult:
    total: int = 0
    identical: int = 0
    differing: int = 0
    missing_source: int = 0
    failed: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.differing == 0 and self.failed == 0 and self.missing_source == 0

    def to_json(self) -> dict:
        return {
            "total": self.total,
            "identical": self.identical,
            "differing": self.differing,
            "missing_source": self.missing_source,
            "failed": self.failed,
            "ok": self.ok,
            "details": self.details,
        }


def verify(
    workspace: str,
    names: NameTable | None = None,
    source: str | None = None,
    limit: int = 50,
    log: Callable[[str], None] = lambda message: None,
) -> VerifyResult:
    """Re-encode every unpacked resource and compare it with the original blobs."""
    names = names or NameTable()
    manifest_path = os.path.join(workspace, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise SrkitError(f"{workspace} is not an srkit workspace")
    manifest = _read_json(manifest_path)
    source_path = source or (manifest.get("source") or {}).get("path")
    repo = Repository.open(source_path) if source_path and os.path.exists(source_path) else None

    result = VerifyResult()
    for record in manifest.get("packages", []):
        package_name = record["name"]
        package_dir = os.path.join(workspace, record["directory"])
        package_json = os.path.join(package_dir, PACKAGE_FILE)
        if not os.path.isfile(package_json):
            continue
        data = _read_json(package_json)
        original_package = None
        if repo is not None:
            try:
                original_package = repo.load_package(package_name, load_blobs=True)
            except SrkitError:
                original_package = None

        for item in data.get("entries", []):
            result.total += 1
            entry = TocEntry(
                file_id=int(item["file_id"]),
                type_id=int(item["type_id"]),
            )
            original = (
                original_package.entry(entry.file_id, entry.type_id)
                if original_package else None
            )
            resource_dir = os.path.join(package_dir, item.get("resource_dir") or "")
            ir_path = os.path.join(resource_dir, RESOURCE_FILE)
            if not os.path.isfile(ir_path):
                continue
            if original is None:
                result.missing_source += 1
                continue
            try:
                toc, gpu, stream = _encode_resource(ir_path, resource_dir, entry, names)
            except SrkitError as error:
                result.failed += 1
                if len(result.details) < limit:
                    result.details.append(
                        {
                            "package": package_name,
                            "file_id": item["file_id"],
                            "type": item.get("type_name"),
                            "status": "error",
                            "error": str(error),
                        }
                    )
                continue

            status = "identical"
            detail: dict[str, Any] = {}
            for label, produced, expected in (
                ("toc", toc, bytes(original.toc_data)),
                ("gpu", gpu, bytes(original.gpu_data)),
                ("stream", stream, bytes(original.stream_data)),
            ):
                if produced == expected:
                    continue
                status = "differs"
                offset = _first_difference(produced, expected)
                detail[f"{label}_diff"] = {
                    "offset": offset,
                    "produced_size": len(produced),
                    "expected_size": len(expected),
                    "produced_bytes": produced[offset:offset + 16].hex(),
                    "expected_bytes": expected[offset:offset + 16].hex(),
                }
            if status == "identical":
                result.identical += 1
            else:
                result.differing += 1
                if len(result.details) < limit:
                    result.details.append(
                        {
                            "package": package_name,
                            "file_id": item["file_id"],
                            "type": item.get("type_name"),
                            "status": status,
                            **detail,
                        }
                    )
        log(f"   {package_name}: "
            f"{result.identical} identical / {result.differing} differing so far")
    return result


def _first_difference(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def clean(path: str) -> None:
    """Remove a workspace directory (helper for --force workflows)."""
    if os.path.isdir(path):
        shutil.rmtree(path)
