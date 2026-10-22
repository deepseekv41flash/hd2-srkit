"""``bones`` resource codec.

Layout (``stingray/bones.py`` in the add-on)::

    u32   name count (= bone hash count)
    u32   LOD level count
    f32   unknown[LOD level count]
    u32   bone name hashes[name count]
    u32   LOD levels[LOD level count]     (index of the last bone of each LOD)
    ...   NUL terminated bone names, in hash order, until EOF

The name list is what turns a ``u32`` bone hash in a mesh into a readable bone
name, so this codec is the key to making unit files understandable.
"""

from __future__ import annotations

from ..container.package import TocEntry
from ..errors import CodecError
from ..util.binary import Stream
from ..util.hashing import murmur32_hash
from ..util.ids import NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["BonesCodec", "codec"]


class BonesCodec(ResourceCodec):
    name = "bones"
    type_ids: tuple[int, ...] = (0x18DEAD01056B72E9,)  # TYPE_BONES

    def decode(self, entry: TocEntry, names: NameTable, options=None) -> DecodedResource:
        stream = Stream.reader(entry.toc_data)
        num_names = stream.uint32()
        num_lod = stream.uint32()
        unknown = [stream.float32() for _ in range(num_lod)]
        hashes = [stream.uint32() for _ in range(num_names)]
        lod_levels = [stream.uint32() for _ in range(num_lod)]

        region = stream.read()  # everything left: names
        raw_names = region.split(b"\x00")
        if raw_names and raw_names[-1] == b"":
            raw_names.pop()
        bone_names = [item.decode("utf-8", "replace") for item in raw_names]

        consumed = sum(len(item) + 1 for item in bone_names)
        trailing = region[consumed:]

        ir = self.base_ir(entry, "bones")
        ir.update(
            {
                "bone_count": num_names,
                "lod_level_count": num_lod,
                "unknown_floats": unknown,
                "lod_levels": lod_levels,
                "trailing_hex": trailing.hex(),
                "bones": [
                    {
                        "index": index,
                        "hash": str(bone_hash),
                        "hash_hex": f"{bone_hash:08X}",
                        "name": bone_names[index] if index < len(bone_names) else "",
                        "name_matches_hash": (
                            index < len(bone_names)
                            and murmur32_hash(bone_names[index].encode("utf-8")) == bone_hash
                        ),
                    }
                    for index, bone_hash in enumerate(hashes)
                ],
            }
        )
        if len(bone_names) != num_names:
            ir["warning"] = (
                f"name count mismatch: header says {num_names}, found {len(bone_names)}"
            )
        return DecodedResource(ir)

    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        bones = ir.get("bones") or []
        hashes = [int(item["hash"]) for item in bones]
        bone_names = [item.get("name", "") for item in bones]
        lod_levels = list(ir.get("lod_levels") or [])
        unknown = list(ir.get("unknown_floats") or [])
        if not lod_levels and unknown:
            lod_levels = [len(hashes)] * len(unknown)
        if len(unknown) != len(lod_levels):
            raise CodecError("bones: unknown_floats and lod_levels must have equal length")

        stream = Stream.writer()
        stream.uint32(len(hashes))
        stream.uint32(len(lod_levels))
        for value in unknown:
            stream.float32(value)
        for value in hashes:
            stream.uint32(value)
        for value in lod_levels:
            stream.uint32(value)
        for name in bone_names:
            stream.write(name.encode("utf-8") + b"\x00")
        trailing = bytes.fromhex(ir.get("trailing_hex", "") or "")
        if trailing:
            stream.write(trailing)
        return stream.getvalue(), b"", b""


codec = register(BonesCodec())
