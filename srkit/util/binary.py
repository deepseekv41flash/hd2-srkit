"""Typed binary stream (little-endian) used by every codec in srkit.

The API intentionally mirrors the ``MemoryStream`` helper of the HD2SDK
Blender add-on (``utils/memoryStream.py``) so the format logic could be
ported and cross-checked 1:1, but this implementation is written from
scratch and has no Blender / bpy dependency.

Design notes
------------
* A single object can be switched between read and write mode, like the
  original, because several Stingray serializers interleave both modes
  (they seek back to patch offsets).
* ``seek()`` past the end is legal while writing: the gap is filled with
  ``0x00`` (this is how the original writer emits alignment padding).
* ``seek()`` past the end while reading raises, so truncated resources
  fail loudly instead of silently returning garbage.
"""

from __future__ import annotations

import struct

__all__ = ["Stream", "StreamError", "align_up", "align_down", "pad_length"]


class StreamError(Exception):
    """Raised on truncated reads and other stream level failures."""


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value // alignment) * alignment


def pad_length(value: int, alignment: int) -> int:
    """Number of filler bytes needed to align ``value``."""
    return align_up(value, alignment) - value


class Stream:
    """Random access, typed, little-endian byte stream."""

    __slots__ = ("Data", "Location", "reading")

    def __init__(self, Data: bytes | bytearray = b"", IOMode: str = "read"):
        self.Location = 0
        self.Data = bytearray(Data)
        self.reading = IOMode == "read"

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def reader(cls, data: bytes | bytearray) -> "Stream":
        return cls(data, "read")

    @classmethod
    def writer(cls) -> "Stream":
        return cls(b"", "write")

    # ------------------------------------------------------------------ #
    # mode / position
    # ------------------------------------------------------------------ #
    def IsReading(self) -> bool:
        return self.reading

    def IsWriting(self) -> bool:
        return not self.reading

    def SetReadMode(self) -> None:
        self.reading = True

    def SetWriteMode(self) -> None:
        self.reading = False

    def tell(self) -> int:
        return self.Location

    def seek(self, Location: int) -> int:
        if Location < 0:
            raise StreamError(f"negative seek target: {Location}")
        if Location > len(self.Data):
            if self.reading:
                raise StreamError(
                    f"seek past end of stream: {Location} > {len(self.Data)}"
                )
            self.Data.extend(bytes(Location - len(self.Data)))
        self.Location = Location
        return self.Location

    def skip(self, Count: int) -> int:
        return self.seek(self.Location + Count)

    def align(self, Alignment: int, fill: int = 0) -> int:
        """Move to the next ``Alignment`` boundary (zero fill while writing)."""
        target = align_up(self.Location, Alignment)
        if target != self.Location:
            if self.reading:
                # Padding bytes are skipped but must exist.
                self.seek(target)
            else:
                if fill:
                    self.write(bytes([fill]) * (target - self.Location))
                else:
                    self.seek(target)
        return self.Location

    def __len__(self) -> int:
        return len(self.Data)

    def getvalue(self) -> bytes:
        return bytes(self.Data)

    # ------------------------------------------------------------------ #
    # raw byte access
    # ------------------------------------------------------------------ #
    def read(self, length: int = -1) -> bytes:
        if length is None or length < 0:
            length = len(self.Data) - self.Location
        end = self.Location + length
        if end > len(self.Data):
            raise StreamError(
                f"reading past end of stream: need {length} byte(s) at "
                f"0x{self.Location:X}, only {len(self.Data) - self.Location} left"
            )
        chunk = bytes(self.Data[self.Location:end])
        self.Location = end
        return chunk

    def write(self, data: bytes | bytearray) -> None:
        if isinstance(data, str):
            raise TypeError("write() expects bytes, not str")
        end = self.Location + len(data)
        if end > len(self.Data):
            self.Data.extend(bytes(end - len(self.Data)))
        self.Data[self.Location:end] = data
        self.Location = end

    # ------------------------------------------------------------------ #
    # scalars
    # ------------------------------------------------------------------ #
    def serialize(self, value, fmt: str, size: int):
        if self.reading:
            return struct.unpack(fmt, self.read(size))[0]
        self.write(struct.pack(fmt, value))
        return value

    def int8(self, value=0):
        return self.serialize(value, "<b", 1)

    def uint8(self, value=0):
        return self.serialize(value, "<B", 1)

    def int16(self, value=0):
        return self.serialize(value, "<h", 2)

    def uint16(self, value=0):
        return self.serialize(value, "<H", 2)

    def int32(self, value=0):
        return self.serialize(value, "<i", 4)

    def uint32(self, value=0):
        return self.serialize(value, "<I", 4)

    def int64(self, value=0):
        return self.serialize(value, "<q", 8)

    def uint64(self, value=0):
        return self.serialize(value, "<Q", 8)

    def float16(self, value=0.0):
        return self.serialize(value, "<e", 2)

    def float32(self, value=0.0):
        return self.serialize(value, "<f", 4)

    def float64(self, value=0.0):
        return self.serialize(value, "<d", 8)

    # ------------------------------------------------------------------ #
    # byte blobs
    # ------------------------------------------------------------------ #
    def bytes(self, value: bytes | bytearray = b"", size: int = -1):
        """Read/write a fixed size blob (mirrors ``MemoryStream.bytes``).

        While writing, ``value`` is used and resized to ``size`` when it does
        not match, exactly like the reference implementation.
        """
        if size is None or size < 0:
            size = len(value)
        if self.reading:
            return bytearray(self.read(size))
        if len(value) != size:
            value = bytes(value[:size]).ljust(size, b"\x00")
        self.write(value)
        return bytearray(value)

    def blob(self, size: int) -> bytes:
        """Read exactly ``size`` bytes (read mode only helper)."""
        return self.read(size)

    def string(self, size: int) -> str:
        return self.read(size).split(b"\x00")[0].decode("utf-8", "replace")

    def cstring(self) -> str:
        """Read a NUL terminated string (read mode) or write one (write mode)."""
        if self.reading:
            start = self.Location
            while True:
                if self.Location >= len(self.Data):
                    raise StreamError("unterminated string")
                if self.Data[self.Location] == 0:
                    break
                self.Location += 1
            raw = bytes(self.Data[start:self.Location])
            self.Location += 1
            return raw.decode("utf-8", "replace")
        raise StreamError("cstring() write requires an explicit value")

    # ------------------------------------------------------------------ #
    # small vectors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize(value, length: int):
        value = list(value)
        if len(value) < length:
            value.extend([0] * (length - len(value)))
        elif len(value) > length:
            value = value[:length]
        return value

    def vec2_float(self, value=(0.0, 0.0)):
        value = self._resize(value, 2)
        return [self.float32(value[0]), self.float32(value[1])]

    def vec3_float(self, value=(0.0, 0.0, 0.0)):
        value = self._resize(value, 3)
        return [self.float32(value[0]), self.float32(value[1]), self.float32(value[2])]

    def vec4_float(self, value=(0.0, 0.0, 0.0, 0.0)):
        value = self._resize(value, 4)
        return [
            self.float32(value[0]),
            self.float32(value[1]),
            self.float32(value[2]),
            self.float32(value[3]),
        ]

    def vec2_half(self, value=(0.0, 0.0)):
        value = self._resize(value, 2)
        return [self.float16(value[0]), self.float16(value[1])]

    def vec3_half(self, value=(0.0, 0.0, 0.0)):
        value = self._resize(value, 3)
        return [self.float16(value[0]), self.float16(value[1]), self.float16(value[2])]

    def vec4_half(self, value=(0.0, 0.0, 0.0, 0.0)):
        value = self._resize(value, 4)
        return [
            self.float16(value[0]),
            self.float16(value[1]),
            self.float16(value[2]),
            self.float16(value[3]),
        ]

    def vec2_uint8(self, value=(0, 0)):
        value = self._resize(value, 2)
        return [self.uint8(value[0]), self.uint8(value[1])]

    def vec3_uint8(self, value=(0, 0, 0)):
        value = self._resize(value, 3)
        return [self.uint8(value[0]), self.uint8(value[1]), self.uint8(value[2])]

    def vec4_uint8(self, value=(0, 0, 0, 0)):
        value = self._resize(value, 4)
        return [
            self.uint8(value[0]),
            self.uint8(value[1]),
            self.uint8(value[2]),
            self.uint8(value[3]),
        ]

    def vec2_uint16(self, value=(0, 0)):
        value = self._resize(value, 2)
        return [self.uint16(value[0]), self.uint16(value[1])]

    def vec4_uint16(self, value=(0, 0, 0, 0)):
        value = self._resize(value, 4)
        return [
            self.uint16(value[0]),
            self.uint16(value[1]),
            self.uint16(value[2]),
            self.uint16(value[3]),
        ]

    def vec2_uint32(self, value=(0, 0)):
        value = self._resize(value, 2)
        return [self.uint32(value[0]), self.uint32(value[1])]

    def vec3_uint32(self, value=(0, 0, 0)):
        value = self._resize(value, 3)
        return [self.uint32(value[0]), self.uint32(value[1]), self.uint32(value[2])]

    def vec4_uint32(self, value=(0, 0, 0, 0)):
        value = self._resize(value, 4)
        return [
            self.uint32(value[0]),
            self.uint32(value[1]),
            self.uint32(value[2]),
            self.uint32(value[3]),
        ]
