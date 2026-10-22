"""Command line interface: ``python -m srkit <command>``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import __version__
from .container.bundle import BundleSet, read_bundle_database, write_bundle_set, write_bundle_database
from .container.dsar import DsarFile, CHUNK_START
from .container import empty_unit as sv_empty_unit
from .container.package import Package
from .container.repo import GPU_SUFFIX, STREAM_SUFFIX, Repository
from .errors import SrkitError
from .resources import codec_names, decode_entry, encode_entry, get_codec, registered_types
from .util.ids import (
    NameTable,
    TYPE_NAMES,
    parse_resource_id,
    type_name,
)
from .util.hashing import murmur32_hash, murmur64_hash
from . import workspace as ws

DEFAULT_CHUNK_SIZE = 0x40000
DEFAULT_MAX_BUNDLE = 0x40000000


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def default_hashlist_dir() -> str:
    """Locate the name/hash tables, also inside a PyInstaller bundle.

    Search order: sidecar next to the executable (so the lists can be refreshed
    without rebuilding), the bundled copy, then the developer layout.
    """
    candidates: list[str] = []

    if getattr(sys, "frozen", False):
        executable_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(executable_dir, "hashlists"))
        candidates.append(os.path.join(executable_dir, "_internal", "hashlists"))
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(os.path.join(meipass, "hashlists"))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(os.path.dirname(here), "hashlists"))
    candidates.append(os.path.join(here, "hashlists"))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return ""


def load_names(args) -> NameTable:
    directory = getattr(args, "hashlists", None) or default_hashlist_dir()
    return NameTable.load(directory) if directory else NameTable()


def parse_type_filter(values: list[str] | None, names: NameTable) -> list[int] | None:
    if not values:
        return None
    result: list[int] = []
    lookup = {name: type_id for type_id, name in TYPE_NAMES.items()}
    lookup.update({names.type_name(type_id): type_id for type_id in TYPE_NAMES})
    for value in values:
        text = value.strip()
        if text in lookup:
            result.append(lookup[text])
        elif text.lower().startswith("0x"):
            result.append(int(text, 16))
        else:
            try:
                result.append(int(text))
            except ValueError:
                raise SrkitError(
                    f"unknown type '{value}' (known: {', '.join(sorted(lookup))})"
                ) from None
    return result


def parse_id_filter(values: list[str] | None, names: NameTable) -> list[int] | None:
    if not values:
        return None
    return [parse_resource_id(value, names) for value in values]


def human_size(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value} B"


class Log:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def __call__(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_info(args) -> int:
    names = load_names(args)
    repo = Repository.open(args.path)
    print(f"source      : {os.path.abspath(args.path)}")
    print(f"layout      : {repo.kind}")
    if repo.kind == "bundled":
        index = repo.bundles.load_index()
        print(f"bundles     : {index.num_bundles} file(s)")
        print(f"index items : {index.num_packages} (packages + companion streams)")
        declared = repo.declared_package_names()
        available = repo.package_names()
        print(f"database    : {len(declared)} name(s), {len(available)} available here")
    print(f"name table  : {len(names.names)} friendly, {len(names.bones)} bone(s)")

    per_type: dict[str, int] = {}
    per_type_bytes: dict[str, int] = {}
    total_entries = 0
    package_count = 0
    for name in repo.package_names():
        package_count += 1
        try:
            package = repo.load_package(name, load_blobs=False)
        except SrkitError as error:
            print(f"  !! {name}: {error}")
            continue
        total_entries += len(package.entries)
        toc, gpu, stream = repo.raw_streams(name)
        per_type_bytes["(toc)"] = per_type_bytes.get("(toc)", 0) + len(toc)
        per_type_bytes["(gpu)"] = per_type_bytes.get("(gpu)", 0) + len(gpu)
        per_type_bytes["(stream)"] = per_type_bytes.get("(stream)", 0) + len(stream)
        for entry in package.entries:
            key = type_name(entry.type_id)
            per_type[key] = per_type.get(key, 0) + 1
            toc_size, gpu_size, stream_size = entry.sizes()
            per_type_bytes[key] = (
                per_type_bytes.get(key, 0)
                + (toc_size + gpu_size + stream_size if toc_size or gpu_size or stream_size
                   else entry.declared_toc_size + entry.declared_gpu_size
                   + entry.declared_stream_size)
            )
    print(f"packages    : {package_count}")
    print(f"entries     : {total_entries}")
    if per_type:
        print("resource types:")
        for key in sorted(per_type, key=lambda item: -per_type[item]):
            size = per_type_bytes.get(key, 0)
            print(f"  {key:<24} {per_type[key]:>8}   {human_size(size)}")
    return 0


def cmd_list(args) -> int:
    names = load_names(args)
    repo = Repository.open(args.path)
    types = parse_type_filter(args.type, names)
    ids = parse_id_filter(args.id, names)
    output = []
    for package_name in (args.package or repo.package_names()):
        try:
            package = repo.load_package(package_name, load_blobs=False)
        except SrkitError as error:
            print(f"!! {package_name}: {error}", file=sys.stderr)
            continue
        for entry in package.entries:
            if types and entry.type_id not in types:
                continue
            if ids and entry.file_id not in ids:
                continue
            output.append(
                {
                    "package": package_name,
                    "file_id": str(entry.file_id),
                    "file_id_hex": f"0x{entry.file_id:016X}",
                    "type_id": str(entry.type_id),
                    "type": type_name(entry.type_id),
                    "name": names.label(entry.file_id),
                    "toc_size": entry.declared_toc_size,
                    "gpu_size": entry.declared_gpu_size,
                    "stream_size": entry.declared_stream_size,
                    "codec": get_codec(entry.type_id).name if get_codec(entry.type_id) else "-",
                }
            )
            if args.limit and len(output) >= args.limit:
                break
        if args.limit and len(output) >= args.limit:
            break

    if args.json:
        print(json.dumps(output, indent=1, ensure_ascii=False))
        return 0
    print(f"{'type':<15} {'file id':<20} {'name':<40} {'toc':>10} {'gpu':>12} {'stream':>12}")
    for item in output:
        print(
            f"{item['type']:<15} {item['file_id_hex']:<20} {item['name'][:40]:<40} "
            f"{item['toc_size']:>10} {item['gpu_size']:>12} {item['stream_size']:>12}"
        )
    print(f"-- {len(output)} entrie(s)")
    return 0


def cmd_unpack(args) -> int:
    names = load_names(args)
    totals = ws.unpack(
        source=args.path,
        out_dir=args.output,
        names=names,
        packages=args.package,
        types=parse_type_filter(args.type, names),
        file_ids=parse_id_filter(args.id, names),
        include_gpu=not args.no_gpu,
        include_stream=not args.no_stream,
        raw=args.raw,
        decode_vertices=args.decode_vertices,
        overwrite=args.force,
        log=Log(args.quiet),
    )
    print(
        f"unpacked {totals['written']}/{totals['entries']} resource(s) from "
        f"{totals['packages']} package(s) into {args.output} "
        f"({human_size(totals['payload_bytes'])} payload, {totals['errors']} error(s))"
    )
    return 0 if totals["errors"] == 0 else 1


def cmd_pack(args) -> int:
    names = load_names(args)
    if args.name and args.package and len(args.package) > 1:
        print("error: --name only works when a single package is packed", file=sys.stderr)
        return 2
    totals = ws.pack(
        workspace=args.workspace,
        out_dir=args.output,
        names=names,
        source=args.source,
        compress=args.compress,
        chunk_size=args.chunk_size,
        packages=args.package,
        output_name=args.name,
        log=Log(args.quiet),
    )
    print(
        f"packed {totals['packages']} package(s), {totals['entries']} entrie(s): "
        f"{totals['reencoded']} re-encoded, {totals['copied']} copied, "
        f"{totals['errors']} error(s) -> {args.output}"
    )
    return 0 if totals["errors"] == 0 else 1


def cmd_verify(args) -> int:
    names = load_names(args)
    result = ws.verify(
        workspace=args.workspace,
        names=names,
        source=args.source,
        limit=args.limit,
        log=Log(args.quiet),
    )
    if args.json:
        print(json.dumps(result.to_json(), indent=1, ensure_ascii=False))
    else:
        print(
            f"verified {result.total} entrie(s): {result.identical} byte identical, "
            f"{result.differing} differing, {result.failed} failed, "
            f"{result.missing_source} without source"
        )
        for detail in result.details:
            line = f"  {detail['status']:<9} {detail['package']}/{detail['file_id']} ({detail.get('type')})"
            for key in ("toc_diff", "gpu_diff", "stream_diff"):
                if key in detail:
                    diff = detail[key]
                    line += (
                        f"\n      {key[:-5]}: first diff at 0x{diff['offset']:X} "
                        f"({diff['produced_size']} vs {diff['expected_size']} bytes)"
                    )
            if "error" in detail:
                line += f": {detail['error']}"
            print(line)
    return 0 if result.ok else 1


def cmd_extract(args) -> int:
    names = load_names(args)
    repo = Repository.open(args.path)
    types = parse_type_filter(args.type, names)
    ids = parse_id_filter([args.id], names)
    wanted = ids[0]
    for package_name in (args.package or repo.package_names()):
        try:
            package = repo.load_package(package_name, load_blobs=True)
        except SrkitError as error:
            print(f"!! {package_name}: {error}", file=sys.stderr)
            continue
        for entry in package.entries:
            if entry.file_id != wanted:
                continue
            if types and entry.type_id not in types:
                continue
            os.makedirs(args.output, exist_ok=True)
            base = os.path.join(args.output, f"{type_name(entry.type_id)}_{entry.file_id:016X}")
            written = []
            for label, blob in (
                ("toc", entry.toc_data), ("gpu", entry.gpu_data), ("stream", entry.stream_data)
            ):
                if not blob and not args.keep_empty:
                    continue
                path = f"{base}.{label}.bin"
                with open(path, "wb") as handle:
                    handle.write(bytes(blob))
                written.append(path)
            if args.decoded:
                codec = get_codec(entry.type_id)
                decoded = codec.decode(entry, names, {"decode_vertices": True})
                with open(f"{base}.json", "w", encoding="utf-8") as handle:
                    json.dump(decoded.ir, handle, indent=1, ensure_ascii=False)
                written.append(f"{base}.json")
                payload_dir = os.path.join(args.output, f"{base}.payloads")
                os.makedirs(payload_dir, exist_ok=True)
                for payload in decoded.payloads:
                    path = os.path.join(payload_dir, payload.name)
                    with open(path, "wb") as handle:
                        handle.write(payload.data)
                    written.append(path)
            print(f"extracted {type_name(entry.type_id)} {entry.file_id:016X} "
                  f"from {package_name}:")
            for path in written:
                print(f"  {path} ({human_size(os.path.getsize(path))})")
            return 0
    print(f"resource {wanted} not found", file=sys.stderr)
    return 1


def cmd_check(args) -> int:
    """Decode + re-encode resources straight from an archive and compare bytes."""
    names = load_names(args)
    repo = Repository.open(args.path)
    if args.type:
        types = parse_type_filter(args.type, names)
    else:
        types = sorted(
            type_id for type_id, codec in registered_types().items()
            if codec != "raw"
        )
    wanted = set(types)
    package_filter = set(args.package or [])

    stats: dict[str, dict] = {}
    details: list[dict] = []
    checked = 0
    done_types: set[int] = set()
    log = Log(args.quiet)

    for package_name in repo.package_names():
        if package_filter and package_name not in package_filter:
            continue
        if args.limit and all(type_id in done_types for type_id in wanted):
            break
        try:
            package = repo.load_package(package_name, load_blobs=False)
        except SrkitError as error:
            log(f"!! {package_name}: {error}")
            continue
        for entry in package.entries:
            if entry.type_id not in wanted or entry.type_id in done_types:
                continue
            key = type_name(entry.type_id)
            bucket = stats.setdefault(key, {"checked": 0, "identical": 0, "different": 0,
                                            "failed": 0, "toc_bytes": 0})
            if args.limit and bucket["checked"] + bucket["failed"] >= args.limit:
                done_types.add(entry.type_id)
                continue
            try:
                full = repo.load_entry(package_name, entry.file_id, entry.type_id)
                decoded = decode_entry(
                    full, names, {"decode_vertices": args.decode_vertices}
                )
                payloads = {item.name: item.data for item in decoded.payloads
                            if not item.view}
                toc, gpu, stream = encode_entry(decoded.ir, payloads, full, names)
            except SrkitError as error:
                bucket["failed"] += 1
                if len(details) < args.details:
                    details.append({"package": package_name, "file_id": f"{entry.file_id:016X}",
                                    "type": key, "status": "error", "error": str(error)})
                continue

            bucket["checked"] += 1
            bucket["toc_bytes"] += len(full.toc_data)
            checked += 1
            status = "identical"
            for label, produced, expected in (
                ("toc", toc, bytes(full.toc_data)),
                ("gpu", gpu, bytes(full.gpu_data)),
                ("stream", stream, bytes(full.stream_data)),
            ):
                if produced == expected:
                    continue
                status = "different"
                if len(details) < args.details:
                    offset = _first_difference(produced, expected)
                    details.append({
                        "package": package_name, "file_id": f"{entry.file_id:016X}",
                        "type": key, "status": "different", "blob": label,
                        "offset": offset, "produced_size": len(produced),
                        "expected_size": len(expected),
                        "produced": produced[offset:offset + 16].hex(),
                        "expected": expected[offset:offset + 16].hex(),
                    })
            bucket["identical" if status == "identical" else "different"] += 1
            if args.progress and checked % args.progress == 0:
                log(f"   ... {checked} 个资源已检查")

    if args.json:
        payload = json.dumps({"stats": stats, "details": details}, indent=1,
                             ensure_ascii=False)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            print(f"报告已写入 {args.json}")
    else:
        print(f"{'类型':<16}{'已检查':>8}{'逐字节一致':>12}{'有差异':>8}{'失败':>8}{'toc 字节':>14}")
        for key in sorted(stats):
            item = stats[key]
            print(f"{key:<16}{item['checked']:>8}{item['identical']:>12}"
                  f"{item['different']:>8}{item['failed']:>8}{item['toc_bytes']:>14}")
        total_identical = sum(item["identical"] for item in stats.values())
        total_different = sum(item["different"] for item in stats.values())
        total_failed = sum(item["failed"] for item in stats.values())
        print(f"\n合计：一致 {total_identical}，差异 {total_different}，失败 {total_failed}")
        for detail in details:
            if detail["status"] == "different":
                print(f"  差异 {detail['type']} {detail['file_id']} "
                      f"({detail['package']}) {detail['blob']} @0x{detail['offset']:X} "
                      f"{detail['produced_size']}B vs {detail['expected_size']}B")
                print(f"      生成 {detail['produced']}")
                print(f"      原始 {detail['expected']}")
            else:
                print(f"  错误 {detail['type']} {detail['file_id']} "
                      f"({detail['package']}): {detail['error']}")
    return 0 if (sum(item["different"] + item["failed"] for item in stats.values()) == 0) else 1


def _first_difference(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def cmd_rebuild_bundles(args) -> int:
    source = Repository.open(args.path)
    packages: dict[str, bytes] = {}
    for name in (args.package or source.package_names()):
        toc, gpu, stream = source.raw_streams(name)
        packages[name] = toc
        if gpu:
            packages[name + GPU_SUFFIX] = gpu
        if stream:
            packages[name + STREAM_SUFFIX] = stream

    index_header_template = None
    bundle_header_template = None
    index_prefix = None
    if source.kind == "bundled":
        with open(os.path.join(source.root, "bundles.nxa"), "rb") as handle:
            index_header_template = handle.read(0x20)
        with open(source.bundles.bundle_path(0), "rb") as handle:
            bundle_header_template = handle.read(0x20)
        index_prefix = bytes(source.bundles.load_index().header_prefix)
    elif args.path.lower().endswith(".nxa"):
        pass

    index = write_bundle_set(
        out_dir=args.output,
        packages=packages,
        chunk_size=args.chunk_size,
        max_bundle_size=args.max_bundle_size,
        bundle_header_template=bundle_header_template,
        index_header_template=index_header_template,
        index_header_prefix=index_prefix,
        compress=not args.store,
    )
    if args.database:
        template = None
        database_path = os.path.join(source.root, "bundle_database.data")
        if os.path.isfile(database_path):
            with open(database_path, "rb") as handle:
                template = handle.read()
        with open(os.path.join(args.output, "bundle_database.data"), "wb") as handle:
            handle.write(
                write_bundle_database(
                    [
                        item.name for item in index.packages
                        if not item.name.endswith(GPU_SUFFIX)
                        and not item.name.endswith(STREAM_SUFFIX)
                    ],
                    template,
                )
            )

    if not args.quiet:
        print(
            f"wrote {index.num_bundles} bundle file(s) and bundles.nxa holding "
            f"{len(index.packages)} package file(s) into {args.output}"
        )
    return 0


def cmd_empty_unit(args) -> int:
    """Extract an empty-unit template, and optionally build a slot-filling patch.

    A patch only overrides the FileIDs it carries; every Unit slot of the
    target that the patch does not carry keeps rendering its stock part.  This
    command takes the smallest Unit of a patch as a template, reports whether
    it is dependency free / non drawing, and (with ``--retarget``) writes a new
    patch carrying one sanitized copy per requested FileID.
    """
    names = load_names(args)
    repo = Repository.open(args.path)
    wanted = set(args.package or [])
    refs = [ref for ref in repo.package_refs() if not wanted or ref.name in wanted]
    if not refs:
        print("error: no package found", file=sys.stderr)
        return 2

    source = refs[0]
    package = repo.load_package(source.name)
    template_id = parse_resource_id(args.file_id, names) if args.file_id else None
    template = sv_empty_unit.extract_template(package, file_id=template_id)
    print(f"template from {source.name}")
    print(f"  source file id : 0x{template.source_file_id:016X}")
    print(f"  blobs          : toc {len(template.toc_data)} B, "
          f"gpu {len(template.gpu_data)} B, stream {len(template.stream_data)} B")
    print(f"  audit          : {template.audit().describe()}")
    sanitized = sv_empty_unit.sanitize_empty_unit(template.toc_data)
    print(f"  after sanitize : {sv_empty_unit.audit_empty_unit(sanitized).describe()}")

    if not args.retarget:
        return 0

    target_ids: list[int] = []
    for text in args.retarget:
        for part in text.split(","):
            if part.strip():
                target_ids.append(parse_resource_id(part.strip(), names))
    if not target_ids:
        print("error: --retarget listed no usable FileID", file=sys.stderr)
        return 2

    out = Package(name=args.name or source.name)
    for file_id in target_ids:
        out.add_entry(template.clone_for(file_id, verbatim=args.verbatim))
    written = repo.save_package(out, out_dir=args.out, name=out.name)
    print(f"wrote {len(target_ids)} empty unit(s) into {args.out}"
          f"{' (verbatim)' if args.verbatim else ' (sanitized)'}")
    for path in written:
        print(f"  {path} ({human_size(os.path.getsize(path))})")
    return 0


def cmd_codecs(args) -> int:
    print(f"srkit {__version__} - registered codecs")
    mapping = registered_types()
    for type_id in sorted(TYPE_NAMES):
        codec = get_codec(type_id)
        dedicated = "yes" if type_id in mapping else "no (raw fallback)"
        print(f"  {type_name(type_id):<16} 0x{type_id:016X}  {dedicated:<18} "
              f"{codec.name if codec else '-'}")
    print(f"codecs: {', '.join(codec_names())}")
    return 0


def cmd_hash(args) -> int:
    for value in args.names:
        if args.utf16:
            data = value.encode("utf-16-le")
        else:
            data = value.encode("utf-8")
        print(
            f"{value:<40} murmur64=0x{murmur64_hash(data):016X} "
            f"murmur32=0x{murmur32_hash(data):08X}"
        )
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _mod_deploy_files(source_dir: str, game_data: str, log=print) -> list[str]:
    """Copy the patch set into the game folder — never overwriting an existing file."""
    data_dir = os.path.join(source_dir, "data") if os.path.isdir(
        os.path.join(source_dir, "data")) else source_dir
    if not os.path.isdir(game_data):
        raise OSError(f"game data folder not found: {game_data}")
    added = []
    for name in sorted(os.listdir(data_dir)):
        if ".patch_" not in name:
            continue
        target = os.path.join(game_data, name)
        if os.path.exists(target):
            log(f"  keep existing: {name}")
            continue
        shutil.copy2(os.path.join(data_dir, name), target)
        log(f"  added: {name}")
        added.append(name)
    if not added:
        log("  nothing to add (already deployed)")
    return added


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srkit",
        description="Standalone unpacker/packer for Stingray engine resources "
                    "(HD2SDK compatible, no Blender required).",
    )
    parser.add_argument("--version", action="version", version=f"srkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target, with_hashlists: bool = True):
        if with_hashlists:
            target.add_argument("--hashlists", help="directory with friendlynames.txt etc.")

    info = sub.add_parser("info", help="describe an archive / game folder")
    info.add_argument("path")
    add_common(info)
    info.set_defaults(func=cmd_info)

    listing = sub.add_parser("list", help="list resources")
    listing.add_argument("path")
    listing.add_argument("--package", action="append", help="package name (repeatable)")
    listing.add_argument("--type", action="append", help="resource type name or id")
    listing.add_argument("--id", action="append", help="file id (decimal, hex or name)")
    listing.add_argument("--limit", type=int, default=0)
    listing.add_argument("--json", action="store_true")
    add_common(listing)
    listing.set_defaults(func=cmd_list)

    unpack = sub.add_parser("unpack", help="extract resources into a workspace")
    unpack.add_argument("path")
    unpack.add_argument("-o", "--output", required=True)
    unpack.add_argument("--package", action="append")
    unpack.add_argument("--type", action="append")
    unpack.add_argument("--id", action="append")
    unpack.add_argument("--no-gpu", action="store_true", help="skip gpu blobs")
    unpack.add_argument("--no-stream", action="store_true", help="skip stream blobs")
    unpack.add_argument("--raw", action="store_true", help="never decode, keep blobs")
    unpack.add_argument("--decode-vertices", action="store_true",
                        help="also write decoded vertex/index dumps (views)")
    unpack.add_argument("--force", action="store_true", help="unpack into a non empty dir")
    unpack.add_argument("--quiet", action="store_true")
    add_common(unpack)
    unpack.set_defaults(func=cmd_unpack)

    pack = sub.add_parser("pack", help="rebuild packages from a workspace")
    pack.add_argument("workspace")
    pack.add_argument("-o", "--output", required=True)
    pack.add_argument("--source", help="original archive (default: recorded in manifest)")
    pack.add_argument("--package", action="append")
    pack.add_argument("--name", help="output package name, e.g. 9ba626afa44a3aa3.patch_0")
    pack.add_argument("--compress", action="store_true", help="wrap output in DSAR containers")
    pack.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    pack.add_argument("--quiet", action="store_true")
    add_common(pack)
    pack.set_defaults(func=cmd_pack)

    verify = sub.add_parser("verify", help="re-encode and compare against the source")
    verify.add_argument("workspace")
    verify.add_argument("--source")
    verify.add_argument("--limit", type=int, default=25)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--quiet", action="store_true")
    add_common(verify)
    verify.set_defaults(func=cmd_verify)

    extract = sub.add_parser("extract", help="extract a single resource")
    extract.add_argument("path")
    extract.add_argument("--id", required=True)
    extract.add_argument("--type", action="append")
    extract.add_argument("--package", action="append")
    extract.add_argument("-o", "--output", default=".")
    extract.add_argument("--decoded", action="store_true", help="also write the decoded JSON")
    extract.add_argument("--keep-empty", action="store_true")
    add_common(extract)
    extract.set_defaults(func=cmd_extract)

    bundles = sub.add_parser("rebuild-bundles", help="re-pack packages into bundles.NN.nxa")
    bundles.add_argument("path", help="loose or bundled archive directory")
    bundles.add_argument("-o", "--output", required=True)
    bundles.add_argument("--package", action="append")
    bundles.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    bundles.add_argument("--max-bundle-size", type=int, default=DEFAULT_MAX_BUNDLE)
    bundles.add_argument("--store", action="store_true", help="do not LZ4 compress chunks")
    bundles.add_argument("--database", action="store_true", help="also write bundle_database.data")
    bundles.add_argument("--quiet", action="store_true")
    add_common(bundles)
    bundles.set_defaults(func=cmd_rebuild_bundles)

    emptyunit = sub.add_parser(
        "emptyunit",
        help="extract an empty-unit template and build slot-filling patches",
    )
    emptyunit.add_argument("path", help="patch / archive that carries a small Unit")
    emptyunit.add_argument("--package", action="append",
                           help="package/archive entry to take the template from")
    emptyunit.add_argument("--file-id",
                           help="FileID of the Unit to use as template (default: smallest)")
    emptyunit.add_argument("--retarget", action="append", metavar="ID",
                           help="FileID(s) to fill with an empty unit (comma separated)")
    emptyunit.add_argument("--verbatim", action="store_true",
                           help="keep the template bytes as-is instead of sanitizing them")
    emptyunit.add_argument("--name", help="name of the package to write")
    emptyunit.add_argument("-o", "--out", default=".")
    add_common(emptyunit)
    emptyunit.set_defaults(func=cmd_empty_unit)

    codecs = sub.add_parser("codecs", help="list resource codecs")
    codecs.set_defaults(func=cmd_codecs)

    check = sub.add_parser(
        "check",
        help="decode + re-encode resources from an archive and compare the bytes",
    )
    check.add_argument("path")
    check.add_argument("--type", action="append", help="resource type (default: all with a codec)")
    check.add_argument("--package", action="append", help="restrict to these packages")
    check.add_argument("--limit", type=int, default=25,
                       help="max resources per type (0 = no limit)")
    check.add_argument("--details", type=int, default=15, help="how many diffs to print")
    check.add_argument("--decode-vertices", action="store_true")
    check.add_argument("--progress", type=int, default=0, help="log every N resources")
    check.add_argument("--json", nargs="?", const="-", default=None,
                       help="write the report as JSON ('-' = stdout)")
    check.add_argument("--quiet", action="store_true")
    add_common(check)
    check.set_defaults(func=cmd_check)

    hash_cmd = sub.add_parser("hash", help="compute Stingray name hashes")
    hash_cmd.add_argument("names", nargs="+")
    hash_cmd.add_argument("--utf16", action="store_true")
    hash_cmd.set_defaults(func=cmd_hash)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SrkitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # missing files, permission problems, full disks, ...
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
