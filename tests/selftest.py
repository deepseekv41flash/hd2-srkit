"""srkit self test: builds synthetic Stingray resources and proves round trips.

Run with::

    python tests/selftest.py            # all checks
    python tests/selftest.py lz4 unit   # only the named checks

The synthetic files follow the layout documented in ``docs/stingray-format.md``
(which was derived from the HD2SDK Blender add-on).  They are built *by hand*
here - not by the code under test - so the round trips are meaningful in both
directions.  Real game data verification is a separate step (``srkit verify``).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import struct
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srkit.container.bundle import BundleSet, write_bundle_set  # noqa: E402
from srkit.container.dsar import DsarFile, CHUNK_CONTINUE, CHUNK_START  # noqa: E402
from srkit.container.package import Package, TocEntry, TocType  # noqa: E402
from srkit.container.stream_metadata import (  # noqa: E402
    POISON_STREAM_SIZE,
    StreamDeclaration,
    resolve_stream_size,
)
from srkit.container.unit_refs import list_unit_refs, rewrite, rewrite_unit  # noqa: E402
from srkit.container.empty_unit import (  # noqa: E402
    EmptyUnitTemplate,
    audit_empty_unit,
    pad_patch,
    sanitize_empty_unit,
)
from srkit.errors import FormatError  # noqa: E402
from srkit.container.repo import Repository  # noqa: E402
from srkit.resources import decode_entry, encode_entry  # noqa: E402
from srkit.resources.texture import DDS_HEADER_SIZE  # noqa: E402
from srkit.util.binary import Stream, align_up  # noqa: E402
from srkit.util.hashing import murmur32_hash, murmur64_hash  # noqa: E402
from srkit.util.ids import (  # noqa: E402
    NameTable,
    TYPE_BONES,
    TYPE_LUA,
    TYPE_MATERIAL,
    TYPE_PHYSICS,
    TYPE_TEXTURE,
    TYPE_UNIT,
)
from srkit.util.lz4block import compress as lz4_compress, decompress as lz4_decompress  # noqa: E402
from srkit import workspace as ws  # noqa: E402

UNIT_VERSION = 10800438
PARENT_MATERIAL = 15356477064658408677  # "basic"

#: material header (136) + texture arrays (2 x 4 + 2 x 8) + 3 variables x 20
MATERIAL_VALUES_OFFSET = 136 + 2 * 4 + 2 * 8 + 3 * 20

TESTS: list[tuple[str, callable]] = []
FAILURES: list[str] = []


def check(name):
    def decorator(function):
        TESTS.append((name, function))
        return function
    return decorator


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_equal(produced, expected, message):
    if produced != expected:
        if isinstance(produced, bytes) and isinstance(expected, bytes):
            offset = next(
                (index for index in range(min(len(produced), len(expected)))
                 if produced[index] != expected[index]),
                min(len(produced), len(expected)),
            )
            raise AssertionError(
                f"{message}: differs at 0x{offset:X} "
                f"(len {len(produced)} vs {len(expected)})\n"
                f"  produced: {produced[offset:offset + 24].hex()}\n"
                f"  expected: {expected[offset:offset + 24].hex()}"
            )
        raise AssertionError(f"{message}: {produced!r} != {expected!r}")


# --------------------------------------------------------------------------- #
# synthetic resource builders
# --------------------------------------------------------------------------- #
def build_dds_header(width=4, height=4, dxgi=71, mips=1, array_size=1) -> bytes:
    header = bytearray(DDS_HEADER_SIZE)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x00081007)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 20, 0)
    struct.pack_into("<I", header, 24, 1)
    struct.pack_into("<I", header, 28, mips)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = b"DX10"
    struct.pack_into("<I", header, 108, 0x1000)
    struct.pack_into("<I", header, 128, dxgi)
    struct.pack_into("<I", header, 132, 3)
    struct.pack_into("<I", header, 136, 0)
    struct.pack_into("<I", header, 140, array_size)
    struct.pack_into("<I", header, 144, 0)
    return bytes(header)


def build_texture(pixel_bytes=256, placement="gpu"):
    toc = Stream.writer()
    toc.uint32(0x1234)      # unknown id
    toc.uint32(0)           # unknown 1
    toc.uint32(0xFFFFFFFF)  # unknown 2
    for level in range(15):
        toc.uint32(0x1000 + level)   # start
        toc.uint32(0x800 + level)    # bytes left
        toc.uint16(max(1, 8 >> level))
        toc.uint16(max(1, 8 >> level))
    toc.write(build_dds_header())
    pixels = bytes((index * 7 + 3) & 0xFF for index in range(pixel_bytes))
    if placement == "gpu":
        return bytes(toc.Data), pixels, b""
    return bytes(toc.Data), b"", pixels


def build_material():
    tex_ids = [0xABCDEF0123456789, 0x1111222233334444]
    variables = [
        # (class, elements, name hash, offset, stride)
        (0, 1, murmur32_hash(b"g_intensity"), 0, 4),
        (3, 1, murmur32_hash(b"g_color"), 16, 4),
        (1, 1, murmur32_hash(b"g_uvscale"), 32, 4),
    ]
    blob = bytearray(48)
    struct.pack_into("<f", blob, 0, 1.5)
    struct.pack_into("<4f", blob, 16, 0.25, 0.5, 0.75, 1.0)
    struct.pack_into("<2f", blob, 32, 2.0, 3.0)

    out = Stream.writer()
    out.write(bytes(range(12)))                       # undat1
    out.uint32(136 + 12 + 20 * len(variables) + len(blob))  # end offset
    out.uint64(0xDEADBEEF)
    out.uint64(PARENT_MATERIAL)
    out.write(bytes(32))
    out.uint32(len(tex_ids))
    out.write(bytes(36))
    out.uint32(len(variables))
    out.write(bytes(12))
    out.uint32(len(blob))
    out.write(bytes(12))
    for index in range(len(tex_ids)):
        out.uint32(index)
    for tex_id in tex_ids:
        out.uint64(tex_id)
    for klass, elements, name_hash, offset, stride in variables:
        out.uint32(klass)
        out.uint32(elements)
        out.uint32(name_hash)
        out.uint32(offset)
        out.uint32(stride)
    out.write(blob)
    return bytes(out.Data), b"", b""


def build_bones():
    names = ["root", "spine_01", "head"]
    hashes = [murmur32_hash(name.encode()) for name in names]
    out = Stream.writer()
    out.uint32(len(names))
    out.uint32(2)
    out.float32(0.5)
    out.float32(1.5)
    for value in hashes:
        out.uint32(value)
    out.uint32(len(names))
    out.uint32(len(names))
    for name in names:
        out.write(name.encode() + b"\x00")
    return bytes(out.Data), b"", b""


def build_light_list() -> bytes:
    light = (
        struct.pack("<II", 0x1111, 0)
        + struct.pack("<3f", 1.0, 0.5, 0.25)
        + struct.pack("<8f", 2.0, 0.5, 6.0, 2.0, 0.1, 0.7, 0.0, 0.4)
        + struct.pack("<5f", 1.0, 2.0, 3.0, 4.0, 5.0)
        + struct.pack("<B3xI", 1, 0)
        + bytes(32)
    )
    return struct.pack("<I", 1) + struct.pack("<3I", 0, 0, 0) + light


def build_transform_info() -> bytes:
    out = bytearray(struct.pack("<I", 2) + bytes(12))
    for index in range(2):
        out += struct.pack("<9f", 1, 0, 0, 0, 1, 0, 0, 0, 1)
        out += struct.pack("<3f", float(index), 0.0, 0.0)
        out += struct.pack("<3f", 1.0, 1.0, 1.0)
        out += struct.pack("<f", 0.0)
    for _ in range(2):
        out += struct.pack("<16f", *[float(value) for value in range(16)])
    for index in range(2):
        out += struct.pack("<HH", index, 0xFFFF)
    for index in range(2):
        out += struct.pack("<I", murmur32_hash(f"bone{index}".encode()))
    return bytes(out)


def build_bone_info() -> bytes:
    block = bytearray(100)
    struct.pack_into("<IIII", block, 0, 1, 16, 80, 84)
    struct.pack_into("<16f", block, 16, *[float(value) for value in range(16)])
    struct.pack_into("<I", block, 80, 0)
    struct.pack_into("<II", block, 84, 1, 12)
    struct.pack_into("<I", block, 96, 0)
    return struct.pack("<II", 1, 8) + bytes(block)


def build_stream_info() -> bytes:
    block = bytearray(448)
    struct.pack_into("<Q", block, 0, 0x1234567890ABCDEF)
    components = [
        struct.pack("<IIIQ", 0, 2, 0, 0xAAAA),   # position vec3_float
        struct.pack("<IIIQ", 1, 30, 0, 0xBBBB),  # normal unk_normal
        struct.pack("<IIIQ", 4, 1, 0, 0xCCCC),   # uv vec2_float
    ]
    for index, blob in enumerate(components):
        block[8 + index * 20:8 + index * 20 + 20] = blob
    block[320] = 0x77  # garbage inside the reserved component region
    struct.pack_into("<Q", block, 328, len(components))
    struct.pack_into("<Q", block, 336, 1)
    struct.pack_into("<Q", block, 344, 0)
    struct.pack_into("<I", block, 352, 3)      # num vertices
    struct.pack_into("<I", block, 356, 24)     # stride
    struct.pack_into("<Q", block, 360, 0)
    struct.pack_into("<Q", block, 368, 0)
    struct.pack_into("<Q", block, 376, 2)
    struct.pack_into("<Q", block, 384, 0)
    struct.pack_into("<I", block, 392, 3)      # num indices
    struct.pack_into("<I", block, 396, 0)      # index type u16
    struct.pack_into("<Q", block, 400, 0)
    struct.pack_into("<Q", block, 408, 0)
    struct.pack_into("<I", block, 416, 0)      # vertex offset
    struct.pack_into("<I", block, 420, 80)     # vertex size
    struct.pack_into("<I", block, 424, 80)     # index offset
    struct.pack_into("<I", block, 428, 6)      # index size
    block[432:448] = bytes(range(16))
    return (
        struct.pack("<I", 1)      # num streams
        + struct.pack("<I", 16)   # stream offset
        + struct.pack("<I", 0x5555)  # per stream unknown (mesh id)
        + struct.pack("<I", 0)    # unknown 2
        + bytes(block)
    )


def build_mesh_info() -> bytes:
    block = bytearray(128)
    struct.pack_into("<Q", block, 0, 0xDEADBEEF)
    for index in range(32):
        block[8 + index] = (index * 3) & 0xFF      # garbage unknown
    struct.pack_into("<I", block, 40, 0x5555)      # mesh id
    struct.pack_into("<I", block, 44, 0)
    struct.pack_into("<I", block, 48, 0)           # transform index
    struct.pack_into("<I", block, 52, 0)
    struct.pack_into("<i", block, 56, 0)           # lod index
    struct.pack_into("<I", block, 60, 0)           # stream index
    for index in range(40):
        block[64 + index] = (index * 5) & 0xFF     # garbage unknown
    struct.pack_into("<I", block, 104, 1)          # num materials
    struct.pack_into("<I", block, 108, 128)        # material offset
    struct.pack_into("<Q", block, 112, 0)
    struct.pack_into("<I", block, 120, 1)          # num sections
    struct.pack_into("<I", block, 124, 132)        # sections offset
    block += struct.pack("<I", 0x77)               # material slot id
    block += struct.pack("<6I", 0, 0, 3, 0, 3, 0)  # section
    return (
        struct.pack("<I", 1)          # num meshes
        + struct.pack("<I", 12)       # mesh offset
        + struct.pack("<I", 0x5555)   # mesh unknown id
        + bytes(block)
    )


def build_unit_gpu() -> bytes:
    vertices = bytearray()
    for index in range(3):
        vertices += struct.pack("<3f", float(index), float(index) * 2, float(index) * 3)
        vertices += struct.pack("<I", 0x12345678 + index)
        vertices += struct.pack("<2f", index * 0.5, index * 0.25)
    expect_equal(len(vertices), 72, "synthetic vertex buffer size")
    indices = struct.pack("<3H", 0, 1, 2)
    return bytes(vertices) + bytes(8) + indices + bytes(10)


def build_unit(include_materials: bool = True, include_ending: bool = True) -> tuple[bytes, bytes]:
    light_list = build_light_list()
    lod_group = b"LODGROUPDATA" + bytes(20)
    transform_info = build_transform_info()
    bone_info = build_bone_info()
    stream_info = build_stream_info()
    mesh_info = build_mesh_info()
    materials = struct.pack("<I", 1) + struct.pack("<I", 0x77) + struct.pack("<Q", 0x1234)
    unreversed2 = b"UNREV2!!"

    offsets: dict[str, int] = {}
    cursor = 0x80
    offsets["light_list"] = cursor
    cursor += len(light_list)
    offsets["lod_group_list"] = cursor
    cursor += len(lod_group)
    offsets["transform_info"] = cursor
    cursor += len(transform_info)
    cursor = align_up(cursor, 16)
    offsets["bone_info"] = cursor
    cursor += len(bone_info)
    cursor = align_up(cursor, 16)
    offsets["stream_info"] = cursor
    cursor += len(stream_info)
    offsets["mesh_info"] = cursor
    cursor += len(mesh_info)
    if include_materials:
        offsets["materials"] = cursor
        cursor += len(materials)
    cursor += len(unreversed2)
    if include_ending:
        offsets["ending"] = cursor
        cursor += 8

    toc = bytearray(cursor)
    toc[offsets["light_list"]:offsets["light_list"] + len(light_list)] = light_list
    toc[offsets["lod_group_list"]:offsets["lod_group_list"] + len(lod_group)] = lod_group
    toc[offsets["transform_info"]:offsets["transform_info"] + len(transform_info)] = transform_info
    toc[offsets["bone_info"]:offsets["bone_info"] + len(bone_info)] = bone_info
    toc[offsets["stream_info"]:offsets["stream_info"] + len(stream_info)] = stream_info
    toc[offsets["mesh_info"]:offsets["mesh_info"] + len(mesh_info)] = mesh_info
    if include_materials:
        toc[offsets["materials"]:offsets["materials"] + len(materials)] = materials
    toc[cursor - len(unreversed2) - (8 if include_ending else 0):
        cursor - (8 if include_ending else 0)] = unreversed2
    if include_ending:
        struct.pack_into("<Q", toc, offsets["ending"], 1)

    # header
    struct.pack_into("<QQQQQ", toc, 0x00, 0x1, 0x2, 0x3, 0x4, 0x5)
    struct.pack_into("<I", toc, 0x28, 0x99)              # unknown 28
    struct.pack_into("<I", toc, 0x2C, UNIT_VERSION)
    struct.pack_into("<I", toc, 0x30, offsets["lod_group_list"])
    struct.pack_into("<I", toc, 0x34, offsets["transform_info"])
    struct.pack_into("<I", toc, 0x38, offsets["light_list"])
    struct.pack_into("<I", toc, 0x3C, 0)                 # pre light list
    struct.pack_into("<I", toc, 0x40, 0)                 # wwise callback
    toc[0x44:0x4C] = b"HDRDATA2"
    struct.pack_into("<I", toc, 0x4C, 0)                 # customization
    struct.pack_into("<I", toc, 0x50, 0)                 # unknown header 1
    struct.pack_into("<I", toc, 0x54, 0)                 # connecting bones
    struct.pack_into("<I", toc, 0x58, offsets["bone_info"])
    struct.pack_into("<I", toc, 0x5C, offsets["stream_info"])
    struct.pack_into("<I", toc, 0x60, offsets.get("ending", 0))
    struct.pack_into("<I", toc, 0x64, offsets["mesh_info"])
    struct.pack_into("<Q", toc, 0x68, 0xF00D)
    struct.pack_into("<I", toc, 0x70, offsets.get("materials", 0))
    toc[0x74:0x80] = b"PADDING12345"[:12]
    return bytes(toc), build_unit_gpu()


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
@check("lz4")
def test_lz4():
    expect_equal(lz4_decompress(b"\x10a"), b"a", "single literal block")
    expect_equal(lz4_decompress(b"\x40abcd\x04\x00"), b"abcdabcd", "match copy")
    expect_equal(lz4_decompress(b""), b"", "empty block")
    random.seed(1234)
    cases = [
        b"",
        b"a",
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        bytes(range(256)) * 4,
        bytes(random.randrange(256) for _ in range(5000)),
        b"abcabcabcabc" * 500,
    ]
    for index, data in enumerate(cases):
        packed = lz4_compress(data)
        expect_equal(lz4_decompress(packed, len(data)), data, f"lz4 roundtrip #{index}")
    expect(len(lz4_compress(b"a" * 4096)) < 4096 // 4, "lz4 should compress repetition")
    try:
        lz4_decompress(b"\x10a\x01\x00")
        raise AssertionError("truncated block should fail")
    except Exception:
        pass


@check("dsar")
def test_dsar():
    big = bytes(range(256)) * 300
    resources = [b"hello world", big, b""]
    data = DsarFile.build(resources, chunk_size=4096)
    parsed = DsarFile.parse(data, "test")
    expect_equal(parsed.decompress_all(), b"hello world" + big, "dsar full payload")
    expect_equal(parsed.read_resource(0), b"hello world", "first resource")
    expect_equal(parsed.read_resource(11), big, "second resource")
    expect(parsed.chunks[0].starts_resource, "first chunk starts a resource")
    expect(not parsed.chunks[0].continues, "single chunk resource does not continue")
    expect(parsed.chunks[1].starts_resource and parsed.chunks[1].continues,
           "first chunk of a split resource sets start+continue (retail layout)")
    expect(parsed.chunks[2].continues and not parsed.chunks[2].starts_resource,
           "following chunks only continue")
    expect(parsed.chunks[-1].continues and not parsed.chunks[-1].starts_resource,
           "last chunk of a split resource only continues")
    expect_equal(parsed.chunks[0].compression_type, 0,
                 "tiny chunk stays stored (LZ4 would grow it)")
    expect_equal(parsed.chunks[1].compression_type, 3, "compressible chunk uses lz4")
    expect_equal(
        len(DsarFile.parse(DsarFile.build([big], chunk_size=4096)).chunks),
        19,
        "chunk count for a 76800 byte resource with 4096 byte chunks",
    )

    stored = DsarFile.parse(DsarFile.build(resources, compress=False), "stored")
    expect_equal(stored.chunks[0].compression_type, 0, "store mode")
    expect_equal(stored.read_resource(0), b"hello world", "stored resource")


@check("package")
def test_package():
    package = Package("testpkg")
    package.unknown = 0x42
    package.unk4_data = bytearray(range(56))
    package.types = [TocType(type_id=TYPE_TEXTURE), TocType(type_id=TYPE_UNIT)]

    texture_toc, texture_gpu, _ = build_texture()
    unit_toc, unit_gpu = build_unit()
    package.add_entry(TocEntry(file_id=0x1111, type_id=TYPE_TEXTURE,
                               toc_data=bytearray(texture_toc),
                               gpu_data=bytearray(texture_gpu)))
    package.add_entry(TocEntry(file_id=0x2222, type_id=TYPE_UNIT,
                               toc_data=bytearray(unit_toc),
                               gpu_data=bytearray(unit_gpu)))

    toc, gpu, stream = package.to_bytes()
    expect(len(toc) >= 256 * 2, "toc padding")

    parsed = Package.parse(toc, gpu, stream, name="testpkg")
    expect_equal(parsed.magic, 0xF0000011, "toc magic")
    expect_equal(len(parsed.entries), 2, "entry count")
    expect_equal(parsed.types[0].type_id, TYPE_TEXTURE, "type order preserved")
    for original, restored in zip(package.entries, parsed.entries):
        expect_equal(restored.file_id, original.file_id, "file id")
        expect_equal(bytes(restored.toc_data), bytes(original.toc_data), "toc blob")
        expect_equal(bytes(restored.gpu_data), bytes(original.gpu_data), "gpu blob")

    toc2, gpu2, stream2 = parsed.to_bytes()
    expect_equal(toc2, toc, "toc re-serialization")
    expect_equal(gpu2, gpu, "gpu re-serialization")
    expect_equal(stream2, stream, "stream re-serialization")

    # header sanity at fixed offsets
    expect_equal(struct.unpack_from("<I", toc, 0)[0], 0xF0000011, "magic offset")
    expect_equal(struct.unpack_from("<I", toc, 4)[0], 2, "type count offset")
    expect_equal(struct.unpack_from("<I", toc, 8)[0], 2, "entry count offset")
    gpu_offsets = [
        struct.unpack_from("<Q", toc, 0x48 + 2 * 0x20 + index * 0x50 + 0x20)[0]
        for index in range(2)
    ]
    expect(all(offset % 64 == 0 for offset in gpu_offsets),
           f"gpu offsets are 64 byte aligned: {gpu_offsets}")
    stream_offsets = [
        struct.unpack_from("<Q", toc, 0x48 + 2 * 0x20 + index * 0x50 + 0x18)[0]
        for index in range(2)
    ]
    expect(all(offset % 64 == 0 for offset in stream_offsets),
           f"stream offsets are 64 byte aligned: {stream_offsets}")


@check("bundles")
def test_bundles():
    toc, gpu, stream = build_texture()
    unit_toc, unit_gpu = build_unit()
    packages = {
        "alpha": toc,
        "alpha.gpu_resources": gpu,
        "beta": unit_toc,
        "beta.gpu_resources": unit_gpu,
    }
    with tempfile.TemporaryDirectory() as folder:
        index = write_bundle_set(folder, packages, chunk_size=512, max_bundle_size=1024)
        expect(index.num_bundles >= 2, f"expected several bundles, got {index.num_bundles}")
        bundle_set = BundleSet(folder)
        expect_equal(sorted(bundle_set.package_names()), sorted(packages), "package names")
        for name, blob in packages.items():
            expect_equal(bundle_set.extract_package(name), blob, f"rebuilt package {name}")

        # fragment metadata must be consistent with the rebuilt size
        for package in index.packages:
            expect_equal(package.size, len(packages[package.name]), "package size")
            total = sum(size for _, size in package.fragments())
            expect_equal(total, package.size, f"fragments cover {package.name}")


@check("texture")
def test_texture_roundtrip():
    toc, gpu, stream = build_texture()
    entry = TocEntry(file_id=1, type_id=TYPE_TEXTURE, toc_data=bytearray(toc),
                     gpu_data=bytearray(gpu), stream_data=bytearray(stream))
    decoded = decode_entry(entry, NameTable())
    expect_equal(decoded.ir["data_placement"], "gpu", "placement")
    expect_equal(decoded.ir["dds"]["dxgi_format_name"], "BC1_UNORM", "dxgi format")
    payloads = {item.name: item.data for item in decoded.payloads}
    expect_equal(len(payloads["texture.dds"]), DDS_HEADER_SIZE + 256, "dds payload size")
    toc2, gpu2, stream2 = encode_entry(decoded.ir, payloads, entry, NameTable())
    expect_equal(toc2, toc, "texture toc roundtrip")
    expect_equal(gpu2, gpu, "texture gpu roundtrip")

    # streamed texture keeps its placement
    toc_s, gpu_s, stream_s = build_texture(placement="stream")
    entry_s = TocEntry(file_id=2, type_id=TYPE_TEXTURE, toc_data=bytearray(toc_s),
                       stream_data=bytearray(stream_s))
    decoded_s = decode_entry(entry_s, NameTable())
    expect_equal(decoded_s.ir["data_placement"], "stream", "stream placement")
    payloads_s = {item.name: item.data for item in decoded_s.payloads}
    toc2_s, gpu2_s, stream2_s = encode_entry(decoded_s.ir, payloads_s, entry_s, NameTable())
    expect_equal(toc2_s, toc_s, "streamed texture toc")
    expect_equal(gpu2_s, b"", "streamed texture stays out of gpu")
    expect_equal(stream2_s, stream_s, "streamed texture stream")

    # editing the DDS must show up in the packed pixels
    edited = bytearray(payloads["texture.dds"])
    edited[-1] ^= 0xFF
    toc3, gpu3, _ = encode_entry(decoded.ir, {"texture.dds": bytes(edited)}, entry, NameTable())
    expect_equal(toc3, toc, "edited texture toc unchanged")
    expect_equal(gpu3[-1], gpu[-1] ^ 0xFF, "edited pixel reached the gpu blob")


@check("material")
def test_material_roundtrip():
    toc, _, _ = build_material()
    entry = TocEntry(file_id=3, type_id=TYPE_MATERIAL, toc_data=bytearray(toc))
    decoded = decode_entry(entry, NameTable())
    expect_equal(decoded.ir["parent_material_id"], str(PARENT_MATERIAL), "parent id")
    expect_equal(decoded.ir["parent_material_template"], "basic", "parent template")
    expect_equal(len(decoded.ir["shader_variables"]), 3, "variable count")
    expect_equal(decoded.ir["shader_variables"][1]["values"],
                 [0.25, 0.5, 0.75, 1.0], "vec4 values")
    payloads = {item.name: item.data for item in decoded.payloads}
    toc2, _, _ = encode_entry(decoded.ir, payloads, entry, NameTable())
    expect_equal(toc2, toc, "material roundtrip")

    # NaN payloads must survive untouched (values_hex anchor)
    nan_toc = bytearray(toc)
    struct.pack_into("<I", nan_toc, MATERIAL_VALUES_OFFSET, 0x7FC00001)
    entry_nan = TocEntry(file_id=4, type_id=TYPE_MATERIAL, toc_data=nan_toc)
    decoded_nan = decode_entry(entry_nan, NameTable())
    payloads_nan = {item.name: item.data for item in decoded_nan.payloads}
    toc3, _, _ = encode_entry(decoded_nan.ir, payloads_nan, entry_nan, NameTable())
    expect_equal(toc3, bytes(nan_toc), "material NaN payload roundtrip")

    # editing a value patches the blob
    edited_ir = json.loads(json.dumps(decoded.ir))
    edited_ir["shader_variables"][0]["values"] = [42.0]
    toc4, _, _ = encode_entry(edited_ir, payloads, entry, NameTable())
    expect_equal(struct.unpack_from("<f", toc4, MATERIAL_VALUES_OFFSET)[0], 42.0,
                 "edited shader variable value")


@check("bones")
def test_bones_roundtrip():
    toc, _, _ = build_bones()
    entry = TocEntry(file_id=5, type_id=TYPE_BONES, toc_data=bytearray(toc))
    decoded = decode_entry(entry, NameTable())
    expect_equal([item["name"] for item in decoded.ir["bones"]],
                 ["root", "spine_01", "head"], "bone names")
    expect(all(item["name_matches_hash"] for item in decoded.ir["bones"]),
           "bone hashes match the murmur32 of the names")
    toc2, _, _ = encode_entry(decoded.ir, {}, entry, NameTable())
    expect_equal(toc2, toc, "bones roundtrip")


@check("ids")
def test_id_parsing():
    """IDs must be copy/pasteable in every form the CLI prints or documents."""
    from srkit.util.ids import parse_resource_id

    expect_equal(parse_resource_id("4097"), 4097, "decimal id")
    expect_equal(parse_resource_id("0x1001"), 0x1001, "0x prefixed id")
    expect_equal(parse_resource_id("0x12CA15252610A40A"), 0x12CA15252610A40A, "long hex id")
    expect_equal(parse_resource_id("12CA15252610A40A"), 0x12CA15252610A40A,
                 "bare hex id as printed by `srkit list`")
    expect_equal(parse_resource_id("9999999999999999"), 9999999999999999,
                 "16 digits stay decimal, not hex")
    names = NameTable()
    names.names[12345] = "some_name"
    names.reverse["some_name"] = 12345
    expect_equal(parse_resource_id("some_name", names), 12345, "friendly name lookup")


@check("unit")
def test_unit_roundtrip():
    toc, gpu = build_unit()
    entry = TocEntry(file_id=6, type_id=TYPE_UNIT, toc_data=bytearray(toc),
                     gpu_data=bytearray(gpu))
    names = NameTable()
    decoded = decode_entry(entry, names, {"decode_vertices": True})
    ir = decoded.ir
    expect_equal(ir["version"], UNIT_VERSION, "version")
    expect_equal(len(ir["lights"]), 1, "light count")
    expect_equal(len(ir["streams"]), 1, "stream count")
    expect_equal(len(ir["meshes"]), 1, "mesh count")
    expect_equal(ir["streams"][0]["vertex_stride"], 24, "vertex stride")
    expect_equal(sorted(ir["opaque_sections"]),
                 ["bone_info", "lod_group_list", "transform_info"], "opaque sections")
    expect_equal(len(ir["transforms"]), 2, "transform view")

    view = [item for item in decoded.payloads if item.name == "mesh0_vertices.json"]
    expect_equal(len(view), 1, "vertex view payload")
    vertices = json.loads(view[0].data)
    expect_equal(len(vertices), 3, "decoded vertices")
    expect_equal(vertices[1]["position"], [1.0, 2.0, 3.0], "decoded position")
    expect_equal(len(vertices[0]["normal"]), 3, "decoded normal")

    payloads = {item.name: item.data for item in decoded.payloads}
    toc2, gpu2, stream2 = encode_entry(ir, payloads, entry, names)
    expect_equal(toc2, toc, "unit toc roundtrip")
    expect_equal(gpu2, gpu, "unit gpu roundtrip")
    expect_equal(stream2, b"", "unit stream roundtrip")


@check("patch-layout")
def test_patch_layout():
    """Retail patch archives lay toc blobs out on 16 byte boundaries."""
    scripts = [b"-- HD2-Addon: mods/test\nreturn 1\n", bytes(range(256)) * 40]
    blobs = [struct.pack("<II", len(script), 2) + script for script in scripts]
    file_ids = [0xB60D0E278E4E4DB2, 0xFFD5847E62A39CE1]

    header_size = 0x48 + 0x20 * 1 + 0x50 * 2
    offsets = []
    cursor = align_up(header_size, 16)
    for blob in blobs:
        offsets.append(cursor)
        cursor = align_up(cursor + len(blob), 16)

    toc = bytearray(cursor)
    struct.pack_into("<IIII", toc, 0, 0xF0000011, 1, 2, 0)
    struct.pack_into("<QQQII", toc, 0x48, 0, TYPE_LUA, 2, 16, 16)
    for index, blob in enumerate(blobs):
        base = 0x68 + index * 0x50
        struct.pack_into(
            "<QQQQQQQIIIIII", toc, base,
            file_ids[index], TYPE_LUA, offsets[index], 0, 0, 0, 0,
            len(blob), 0, 0, 16, 16, 0,          # entry index stays 0
        )
        toc[offsets[index]:offsets[index] + len(blob)] = blob

    package = Package.parse(bytes(toc), name="patch")
    expect_equal(package.entries[0].entry_index, 0, "entry index 0 is preserved")
    toc2, _, _ = package.to_bytes()
    expect_equal(toc2, bytes(toc), "retail patch layout round trip")
    expect_equal(len(toc2) % 16, 0, "toc padded to 16 bytes")
    for entry in package.entries:
        expect_equal(entry.toc_data_offset % 16, 0, "toc blob offset is 16 aligned")

    # the same package through the workspace pipeline
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "patch_0")
        with open(source, "wb") as handle:
            handle.write(bytes(toc))
        unpacked = os.path.join(folder, "ws")
        ws.unpack(source, unpacked)
        result = ws.verify(unpacked)
        expect_equal(result.differing, 0, f"patch verify: {result.details}")
        packed = os.path.join(folder, "out")
        ws.pack(unpacked, packed, output_name="9ba626afa44a3aa3.patch_0")
        with open(os.path.join(packed, "9ba626afa44a3aa3.patch_0"), "rb") as handle:
            expect_equal(handle.read(), bytes(toc), "packed patch file")


@check("lua")
def test_lua():
    samples = [
        (b"-- HD2-Addon: mods/test\nreturn require('x')\n", "script.lua"),
        (b"\x1bLJ\x02\x02" + bytes(range(64)) * 4, "script.bin"),
    ]
    for payload, expected_name in samples:
        toc = struct.pack("<II", len(payload), 2) + payload
        entry = TocEntry(file_id=9, type_id=TYPE_LUA, toc_data=bytearray(toc))
        decoded = decode_entry(entry, NameTable())
        payloads = {item.name: item.data for item in decoded.payloads}
        expect(expected_name in payloads,
               f"{expected_name} payload expected, got {sorted(payloads)}")
        expect_equal(payloads[expected_name], payload, "lua payload content")
        toc2, _, _ = encode_entry(decoded.ir, payloads, entry, NameTable())
        expect_equal(toc2, toc, "lua roundtrip")

    # editing the script changes exactly the payload
    payload = samples[0][0]
    toc = struct.pack("<II", len(payload), 2) + payload
    entry = TocEntry(file_id=9, type_id=TYPE_LUA, toc_data=bytearray(toc))
    decoded = decode_entry(entry, NameTable())
    edited = b"-- edited\nreturn 42\n"
    toc2, _, _ = encode_entry(decoded.ir, {"script.lua": edited}, entry, NameTable())
    expect_equal(toc2, struct.pack("<II", len(edited), 2) + edited, "edited lua script")


@check("real-variants")
def test_real_world_variants():
    """Regressions found while validating against retail Helldivers 2 data."""
    names = NameTable()

    # 1) a bundle fragment may span several consecutive resources
    first, second = b"A" * 40, b"B" * 70
    dsar = DsarFile.parse(DsarFile.build([first, second], chunk_size=10, compress=False))
    expect_equal(dsar.read_resource_span(0, 110), first + second, "span two resources")
    expect_equal(dsar.read_resource_span(40, 70), second, "second resource by offset")
    expect_equal(dsar.read_stream_range(45, 20), second[5:25],
                 "arbitrary offset inside the stream")
    expect(dsar.chunks[0].starts_resource and dsar.chunks[0].continues,
           "first chunk of a split resource is start+continue")
    expect(all(not chunk.starts_resource for chunk in dsar.chunks[1:4]),
           "following chunks never start a resource")

    # 2) retail streamed textures carry a gpu blob *and* a stream blob
    toc, gpu, stream = build_texture(pixel_bytes=64, placement="stream")
    entry = TocEntry(file_id=0x21, type_id=TYPE_TEXTURE, toc_data=bytearray(toc),
                     gpu_data=bytearray(b"\x5A" * 96), stream_data=bytearray(stream))
    decoded = decode_entry(entry, names)
    payloads = {item.name: item.data for item in decoded.payloads}
    expect("gpu.bin" in payloads, "the untouched gpu blob is preserved")
    expect_equal(decoded.ir["data_placement"], "stream", "stream holds the pixels")
    toc2, gpu2, stream2 = encode_entry(decoded.ir, payloads, entry, names)
    expect_equal((toc2, gpu2, stream2), (toc, bytes(entry.gpu_data), stream),
                 "texture with two blobs round trips")

    # 3) retail materials can carry a gpu blob the codec does not interpret
    material_toc, _, _ = build_material()
    entry = TocEntry(file_id=0x22, type_id=TYPE_MATERIAL,
                     toc_data=bytearray(material_toc), gpu_data=bytearray(b"\x11\x22" * 200))
    decoded = decode_entry(entry, names)
    payloads = {item.name: item.data for item in decoded.payloads}
    expect("blob.gpu.bin" in payloads, "unclaimed blobs are preserved by the framework")
    toc2, gpu2, _ = encode_entry(decoded.ir, payloads, entry, names)
    expect_equal(toc2, material_toc, "material toc round trips")
    expect_equal(gpu2, bytes(entry.gpu_data), "material gpu blob round trips")

    # 4) units exist without a materials section and without an end marker
    for include_materials, include_ending in ((True, True), (False, True),
                                              (True, False), (False, False)):
        toc, gpu = build_unit(include_materials=include_materials,
                             include_ending=include_ending)
        entry = TocEntry(file_id=0x23, type_id=TYPE_UNIT, toc_data=bytearray(toc),
                         gpu_data=bytearray(gpu))
        decoded = decode_entry(entry, names)
        payloads = {item.name: item.data for item in decoded.payloads}
        toc2, gpu2, _ = encode_entry(decoded.ir, payloads, entry, names)
        expect_equal(toc2, toc,
                     f"unit round trip (materials={include_materials}, ending={include_ending})")

    # 5) a retail toc blob may start well past the 16 byte alignment
    package = Package("retail")
    package.types = [TocType(type_id=TYPE_TEXTURE)]
    bodies = [bytes([index]) * (100 + index * 7) for index in range(3)]
    cursor = 0x400
    for index, body in enumerate(bodies):
        entry = TocEntry(file_id=0x30 + index, type_id=TYPE_TEXTURE,
                         toc_data=bytearray(body), toc_data_offset=cursor)
        cursor = align_up(cursor + len(body), 16)
        package.add_entry(entry)
    toc, _, _ = package.to_bytes()
    expect_equal(struct.unpack_from("<Q", toc, 0x48 + 0x20 + 0x10)[0], 0x400,
                 "the original 0x400 blob offset is kept")
    parsed = Package.parse(toc, name="retail")
    toc2, _, _ = parsed.to_bytes()
    expect_equal(toc2, toc, "package with large first padding round trips")

    # 6) stream blobs are appended without extra alignment
    package = Package("streams")
    package.types = [TocType(type_id=TYPE_TEXTURE)]
    for index in range(3):
        entry = TocEntry(file_id=0x40 + index, type_id=TYPE_TEXTURE,
                         toc_data=bytearray(b"\x01" * 32),
                         stream_data=bytearray(b"\x02" * 100))
        package.add_entry(entry)
    toc, _, stream = package.to_bytes()
    offsets = [struct.unpack_from("<Q", toc, 0x48 + 0x20 + i * 0x50 + 0x18)[0]
               for i in range(3)]
    expect_equal(offsets, [0, 100, 200], f"stream offsets are contiguous: {offsets}")
    expect_equal(len(stream), 300, "stream blob size")


@check("workspace")
def test_workspace():
    texture_toc, texture_gpu, _ = build_texture()
    unit_toc, unit_gpu = build_unit()
    material_toc, _, _ = build_material()
    bones_toc, _, _ = build_bones()
    physics_toc = bytes(random.Random(7).randrange(256) for _ in range(64))

    package = Package("9ba626afa44a3aa3")
    package.types = [
        TocType(type_id=TYPE_TEXTURE),
        TocType(type_id=TYPE_UNIT),
        TocType(type_id=TYPE_MATERIAL),
        TocType(type_id=TYPE_BONES),
        TocType(type_id=TYPE_PHYSICS),
    ]
    package.add_entry(TocEntry(file_id=0x1001, type_id=TYPE_TEXTURE,
                               toc_data=bytearray(texture_toc), gpu_data=bytearray(texture_gpu)))
    package.add_entry(TocEntry(file_id=0x1002, type_id=TYPE_UNIT,
                               toc_data=bytearray(unit_toc), gpu_data=bytearray(unit_gpu)))
    package.add_entry(TocEntry(file_id=0x1003, type_id=TYPE_MATERIAL,
                               toc_data=bytearray(material_toc)))
    package.add_entry(TocEntry(file_id=0x1004, type_id=TYPE_BONES,
                               toc_data=bytearray(bones_toc)))
    package.add_entry(TocEntry(file_id=0x1005, type_id=TYPE_PHYSICS,
                               toc_data=bytearray(physics_toc)))

    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "source")
        os.makedirs(source)
        toc, gpu, stream = package.to_bytes()
        for suffix, blob in (("", toc), (".gpu_resources", gpu), (".stream", stream)):
            if blob:
                with open(os.path.join(source, "9ba626afa44a3aa3" + suffix), "wb") as handle:
                    handle.write(blob)

        repo = Repository.open(source)
        expect_equal(repo.kind, "loose", "loose layout detected")
        expect_equal(repo.package_names(), ["9ba626afa44a3aa3"], "package discovery")

        unpacked = os.path.join(folder, "unpacked")
        totals = ws.unpack(source, unpacked, decode_vertices=True)
        expect_equal(totals["written"], 5, "all entries unpacked")
        expect(os.path.isfile(os.path.join(unpacked, ws.MANIFEST_NAME)), "manifest written")

        result = ws.verify(unpacked)
        expect_equal(result.differing, 0, f"no differing resources: {result.details}")
        expect_equal(result.failed, 0, f"no failed resources: {result.details}")
        expect_equal(result.identical, 5, "every resource byte identical")

        # a raw workspace must round trip too (raw codec is chosen from the IR)
        raw_ws = os.path.join(folder, "raw_unpacked")
        ws.unpack(source, raw_ws, raw=True)
        raw_result = ws.verify(raw_ws)
        expect_equal(raw_result.differing, 0, f"raw verify: {raw_result.details}")
        expect_equal(raw_result.identical, 5, "raw workspace byte identical")
        raw_packed = os.path.join(folder, "raw_packed")
        ws.pack(raw_ws, raw_packed)
        with open(os.path.join(raw_packed, "9ba626afa44a3aa3"), "rb") as handle:
            expect_equal(handle.read(), toc, "raw repack matches the source toc")

        packed = os.path.join(folder, "packed")
        ws.pack(unpacked, packed)
        for suffix, blob in (("", toc), (".gpu_resources", gpu), (".stream", stream)):
            if not blob:
                continue
            with open(os.path.join(packed, "9ba626afa44a3aa3" + suffix), "rb") as handle:
                expect_equal(handle.read(), blob, f"repacked {suffix or 'toc'}")

        # edit a texture and make sure only that resource changes
        texture_dds = os.path.join(
            unpacked, "9ba626afa44a3aa3", "resources", "texture_0000000000001001", "texture.dds"
        )
        with open(texture_dds, "r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            last = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([last[0] ^ 0xFF]))
        edited_dir = os.path.join(folder, "edited")
        ws.pack(unpacked, edited_dir)
        with open(os.path.join(edited_dir, "9ba626afa44a3aa3.gpu_resources"), "rb") as handle:
            edited_gpu = handle.read()
        # the texture gpu blob sits at offset 0, the unit gpu blob behind it
        expect_equal(edited_gpu[:len(texture_gpu) - 1], texture_gpu[:-1],
                     "rest of the gpu blob untouched")
        expect_equal(edited_gpu[len(texture_gpu) - 1], texture_gpu[-1] ^ 0xFF,
                     "edit reached the packed gpu blob")
        expect_equal(edited_gpu[len(texture_gpu):], gpu[len(texture_gpu):],
                     "other resources untouched")
        with open(os.path.join(edited_dir, "9ba626afa44a3aa3"), "rb") as handle:
            expect_equal(handle.read(), toc, "toc unchanged by a texture edit")


@check("bundled-workspace")
def test_bundled_workspace():
    texture_toc, texture_gpu, _ = build_texture(pixel_bytes=1024)
    unit_toc, unit_gpu = build_unit()
    packages = {
        "9ba626afa44a3aa3": texture_toc + unit_toc,  # not a real toc; only layout matters
        "9ba626afa44a3aa3.gpu_resources": texture_gpu + unit_gpu,
    }
    with tempfile.TemporaryDirectory() as folder:
        write_bundle_set(folder, packages, chunk_size=1024, max_bundle_size=4096)
        repo = Repository.open(folder)
        expect_equal(repo.kind, "bundled", "bundled layout detected")
        expect_equal(sorted(repo.bundles.package_names()), sorted(packages),
                     "index lists packages and their companion streams")
        expect_equal(repo.package_names(), ["9ba626afa44a3aa3"],
                     "package_names() hides companion streams")
        for name, blob in packages.items():
            expect_equal(repo.raw_streams(name)[0], blob, f"bundle payload {name}")


@check("stream-metadata")
def test_stream_metadata():
    """The two stream sizes HD2 declares wrong must be repaired, not trusted."""
    unit_toc, _unit_gpu = build_unit()
    expect_equal(POISON_STREAM_SIZE, 0xDFA64C92, "poison sentinel value")

    resolved = resolve_stream_size(
        StreamDeclaration(file_id=1, type_id=TYPE_UNIT, offset=0, size=1234, toc_data=unit_toc), 0)
    expect(resolved.repair is not None, "a Unit that claims a stream size must be repaired")
    expect_equal(resolved.size, 0, "Unit stream size after repair")

    resolved = resolve_stream_size(
        StreamDeclaration(file_id=2, type_id=TYPE_BONES, offset=0, size=8, toc_data=b""), 0)
    expect_equal(resolved.size, 0, "Bones stream size after repair")

    texture_toc = bytearray(0x40)
    struct.pack_into("<I", texture_toc, 0x10, 777)
    # the size in the table is the *literal* sentinel, not our constant: if the
    # constant is edited the detection has to fail
    resolved = resolve_stream_size(
        StreamDeclaration(file_id=3, type_id=TYPE_TEXTURE, offset=0,
                          size=0xDFA64C92, toc_data=bytes(texture_toc)), 4096)
    expect(resolved.repair is not None and resolved.repair.poison_value_detected,
           "the poison sentinel must be flagged")
    expect_equal(resolved.size, 777, "Texture stream size comes from the first mip BytesLeft")

    try:
        resolve_stream_size(
            StreamDeclaration(file_id=4, type_id=TYPE_TEXTURE, offset=0,
                              size=0xDFA64C92, toc_data=b"\x00" * 8), 64)
    except FormatError:
        pass
    else:
        raise AssertionError("a texture toc too short to hold BytesLeft must raise")

    resolved = resolve_stream_size(
        StreamDeclaration(file_id=5, type_id=TYPE_TEXTURE, offset=0, size=16,
                          toc_data=bytes(texture_toc)), 64)
    expect(resolved.repair is None, "an in-bounds declaration must be left alone")

    # parsing applies the repair and reports it
    package = Package(name="stream-test")
    entry = TocEntry(file_id=0x1234, type_id=TYPE_UNIT)
    entry.set_data(toc=unit_toc, gpu=_unit_gpu)
    package.add_entry(entry)
    toc_bytes, gpu_bytes, _stream_bytes = package.to_bytes()
    declared = bytearray(toc_bytes)
    entries_start = 0x48 + len(package.types) * 0x20
    struct.pack_into("<I", declared, entries_start + 60, 4321)   # lie about the stream size
    parsed = Package.parse(bytes(declared), gpu_bytes, b"", name="stream-test")
    expect_equal(len(parsed.stream_repairs), 1, "exactly one stream repair must be reported")
    expect_equal(parsed.stream_repairs[0].repaired_size, 0, "reported repair size")
    expect_equal(len(parsed.entries[0].stream_data), 0, "Unit must end up with no stream payload")


@check("unit-refs")
def test_unit_refs():
    """Unit FileID / SlotID rewriting touches only the reference fields."""
    toc, _gpu = build_unit()
    refs = list_unit_refs(toc)
    expect_equal(refs[:5], [1, 2, 3, 4, 5], "unit header refs")
    expect_equal(refs[5], 0x1234, "unit material id")

    rewritten = rewrite_unit(toc, {2: 0xBBBB, 0x1234: 0xAAAA}, {0x77: 0x88})
    refs = list_unit_refs(rewritten)
    expect_equal(refs[1], 0xBBBB, "bones ref must be remapped")
    expect_equal(refs[5], 0xAAAA, "material id must be remapped")
    materials_offset = struct.unpack_from("<I", rewritten, 0x70)[0]
    expect_equal(struct.unpack_from("<I", rewritten, materials_offset + 4)[0], 0x88,
                 "slot id must be remapped")
    expect_equal(rewrite(TYPE_TEXTURE, b"abc", {1: 2}), b"abc",
                 "unknown resource types must be returned untouched")


@check("empty-unit")
def test_empty_unit():
    """Sanitizing kills the refs and indices; padding fills uncovered slots."""
    toc, gpu = build_unit()
    audit = audit_empty_unit(toc)
    expect_equal(audit.material_ids, [0x1234], "material id detected")
    expect(audit.num_indices > 0, "the synthetic unit must have indices")
    expect(not audit.is_non_drawing(), "the synthetic unit must draw")

    sanitized = sanitize_empty_unit(toc)
    clean = audit_empty_unit(sanitized)
    expect_equal(clean.material_ids, [0], "material refs must be zeroed")
    expect_equal(clean.num_indices, 0, "index counts must be zeroed")
    expect(clean.is_non_drawing(), "the sanitized unit must be non drawing")

    template = EmptyUnitTemplate.from_parts("t", toc, gpu, source_file_id=1)
    entry = template.clone_for(0xDEADBEEF)
    expect_equal(entry.file_id, 0xDEADBEEF, "retargeted file id")
    expect_equal(entry.type_id, TYPE_UNIT, "cloned entry must be a Unit")
    expect_equal(bytes(entry.toc_data), sanitized, "clone carries the sanitized blob")
    expect_equal(bytes(entry.gpu_data), gpu, "gpu bytes are copied verbatim")

    package = Package(name="pad-test")
    package.add_entry(template.clone_for(0x1111))
    added = pad_patch(package, [0x1111, 0x2222], template, mode="sanitized")
    expect_equal(added, [0x2222], "only uncovered slots are filled")
    expect_equal(len(package.entries_of_type(TYPE_UNIT)), 2, "unit count after padding")

    toc_bytes, gpu_bytes, _stream_bytes = package.to_bytes()
    parsed = Package.parse(toc_bytes, gpu_bytes, b"", name="pad-test")
    expect_equal(sorted(item.file_id for item in parsed.entries), [0x1111, 0x2222],
                 "both slots must survive the round trip")
    expect_equal(bytes(parsed.entry(0x2222, TYPE_UNIT).gpu_data), gpu,
                 "padded gpu bytes must survive the round trip")


def main(argv: list[str]) -> int:
    selected = set(argv) if argv else None
    for name, function in TESTS:
        if selected and name not in selected:
            continue
        try:
            function()
        except Exception:  # noqa: BLE001
            FAILURES.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"ok    {name}")
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
