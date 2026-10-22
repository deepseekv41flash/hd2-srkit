"""``lua`` resource codec.

Layout (derived from a retail patch archive - ``9ba626afa44a3aa3.patch_0``,
which holds a Lua mod)::

    u32 payload size
    u32 unknown (2 in every observed sample)
    ... payload (UTF-8 Lua source *or* LuaJIT bytecode, magic b"\\x1bLJ")
    ... optional trailing bytes

The payload is exposed as an editable ``script.lua`` / ``script.bin`` payload, so
a mod's script can be replaced without touching anything else.
"""

from __future__ import annotations

from ..container.package import TocEntry
from ..errors import CodecError
from ..util.binary import Stream
from ..util.ids import NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["LuaCodec", "codec", "LUAJIT_MAGIC", "HEADER_SIZE"]

LUAJIT_MAGIC = b"\x1bLJ"
HEADER_SIZE = 8


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    if data[:3] == LUAJIT_MAGIC:
        return False
    sample = data[:512]
    printable = sum(
        1 for byte in sample
        if byte in (9, 10, 13) or 32 <= byte < 127 or byte >= 128
    )
    return printable >= len(sample) * 0.95


class LuaCodec(ResourceCodec):
    name = "lua"
    type_ids: tuple[int, ...] = (0xA14E8DFA2CD117E2,)  # TYPE_LUA

    def decode(self, entry: TocEntry, names: NameTable, options=None) -> DecodedResource:
        data = bytes(entry.toc_data)
        if len(data) < HEADER_SIZE:
            raise CodecError(f"lua: toc blob too small ({len(data)} bytes)")
        stream = Stream.reader(data)
        size = stream.uint32()
        unknown = stream.uint32()
        if size > len(data) - HEADER_SIZE:
            raise CodecError(
                f"lua: declared payload size {size} exceeds the blob "
                f"({len(data) - HEADER_SIZE} bytes available)"
            )
        payload = data[HEADER_SIZE:HEADER_SIZE + size]
        trailing = data[HEADER_SIZE + size:]
        is_bytecode = payload[:3] == LUAJIT_MAGIC
        is_text = _looks_like_text(payload)

        ir = self.base_ir(entry, "lua")
        ir.update(
            {
                "payload_size": size,
                "unknown": unknown,
                "content": "luajit_bytecode" if is_bytecode else ("text" if is_text else "binary"),
                "trailing_bytes": len(trailing),
                "preview": (
                    payload[:400].decode("utf-8", "replace") if is_text else ""
                ),
            }
        )
        name = "script.lua" if is_text else "script.bin"
        payloads = [
            Payload(
                name,
                payload,
                editable=True,
                note="Lua source" if is_text else "LuaJIT bytecode",
            )
        ]
        if trailing:
            payloads.append(self.master_payload("trailing.bin", trailing))
        return DecodedResource(ir, payloads)

    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        payload = payloads.get("script.lua")
        if payload is None:
            payload = payloads.get("script.bin")
        if payload is None:
            payload = bytes(entry.toc_data)[HEADER_SIZE:]
        stream = Stream.writer()
        stream.uint32(len(payload))
        stream.uint32(int(ir.get("unknown", 2)))
        stream.write(payload)
        trailing = payloads.get("trailing.bin", b"")
        if trailing:
            stream.write(trailing)
        return stream.getvalue(), b"", b""


codec = register(LuaCodec())
