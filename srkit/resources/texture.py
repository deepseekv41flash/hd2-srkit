"""``texture`` resource codec: Stingray texture header + DDS payload.

Layout (``stingray/texture.py`` in the add-on)::

    u32   unknown id
    u32   unknown 1
    u32   unknown 2
    15 x mipmap info, 12 bytes each
            u32 start, u32 bytes left, u16 height, u16 width
    148   bytes DDS header (must carry the DX10 extended header)
    ...   raw texture data

The pixel data does *not* live in the toc blob: it is either in the gpu
resource (most textures) or in the stream resource (streamed ones).  The add-on
always writes it back into the gpu blob, which silently breaks streamed
textures; this codec records the original placement and puts it back there.

Payload ``texture.dds`` is editable: replace it with any DX10 header DDS (BC
formats) and pack again.
"""

from __future__ import annotations

import hashlib
import struct

from ..container.package import TocEntry
from ..errors import CodecError
from ..util.binary import Stream
from ..util.ids import NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["TextureCodec", "codec", "DXGI_FORMATS", "dxgi_format_name", "dxgi_block_size"]

DDS_HEADER_SIZE = 148
DDS_MAGIC = b"DDS "
DX10_MAGIC = b"DX10"

#: only the formats that matter for HD2 + the ones the add-on knows about
DXGI_FORMATS = {
    0: "UNKNOWN",
    10: "R16G16B16A16_FLOAT",
    11: "R16G16B16A16_UNORM",
    28: "R8G8B8A8_UNORM",
    29: "R8G8B8A8_UNORM_SRGB",
    41: "R32_FLOAT",
    49: "R8G8_UNORM",
    61: "R8_UNORM",
    70: "BC1_TYPELESS",
    71: "BC1_UNORM",
    72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM",
    75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM",
    78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM",
    81: "BC4_SNORM",
    83: "BC5_UNORM",
    84: "BC5_SNORM",
    87: "B8G8R8A8_UNORM",
    95: "BC6H_UF16",
    96: "BC6H_SF16",
    98: "BC7_UNORM",
    99: "BC7_UNORM_SRGB",
}


def dxgi_format_name(value: int) -> str:
    return DXGI_FORMATS.get(int(value), f"DXGI_{int(value)}")


def dxgi_block_size(value: int) -> int:
    """Bytes per 4x4 block (BC formats) - used for informational mip sizing."""
    name = dxgi_format_name(value)
    if name.startswith(("BC1", "BC4")):
        return 8
    if name.startswith("BC"):
        return 16
    return 0


class TextureCodec(ResourceCodec):
    name = "texture"
    type_ids: tuple[int, ...] = (0xCD4238C6A0C69E32,)  # TYPE_TEXTURE
    handles = ("toc", "gpu", "stream")

    def decode(self, entry: TocEntry, names: NameTable, options=None) -> DecodedResource:
        stream = Stream.reader(entry.toc_data)
        unk_id = stream.uint32()
        unk1 = stream.uint32()
        unk2 = stream.uint32()
        mipmaps = []
        for _ in range(15):
            mipmaps.append(
                {
                    "start": stream.uint32(),
                    "bytes_left": stream.uint32(),
                    "height": stream.uint16(),
                    "width": stream.uint16(),
                }
            )
        dds_header = bytes(stream.bytes(b"", DDS_HEADER_SIZE))
        remaining = bytes(stream.read()) if stream.tell() < len(stream.Data) else b""

        if len(entry.stream_data) > 0:
            placement = "stream"
            pixel_data = bytes(entry.stream_data)
        elif len(entry.gpu_data) > 0:
            placement = "gpu"
            pixel_data = bytes(entry.gpu_data)
        else:
            placement = "toc"
            pixel_data = remaining

        ir = self.base_ir(entry, "texture")
        ir.update(
            {
                "unknown_id": str(unk_id),
                "unknown1": str(unk1),
                "unknown2": str(unk2),
                "data_placement": placement,
                "mipmaps": mipmaps,
                "dds": self._describe_dds(dds_header),
                "pixel_data_size": len(pixel_data),
                "dds_header_hex": dds_header.hex(),
                "trailing_toc_bytes": len(remaining),
                "gpu_size": len(entry.gpu_data),
                "stream_size": len(entry.stream_data),
            }
        )
        payloads = [
            Payload(
                "texture.dds",
                dds_header + pixel_data,
                editable=True,
                note="DDS (DX10 header) - replace to change the texture",
            )
        ]
        if remaining:
            # keeps a byte faithful round trip if the toc blob had extra bytes
            payloads.append(self.master_payload("toc_tail.bin", remaining))
        # Retail streamed textures carry *both* a gpu and a stream blob; only the
        # stream blob holds the mip pyramid the add-on reads.  Everything that is
        # not the editable pixel source is preserved verbatim.
        if placement != "gpu" and entry.gpu_data:
            payloads.append(
                self.master_payload("gpu.bin", bytes(entry.gpu_data),
                                    note="gpu blob kept verbatim (not the pixel source)")
            )
        if placement != "stream" and entry.stream_data:
            payloads.append(
                self.master_payload("stream.bin", bytes(entry.stream_data),
                                    note="stream blob kept verbatim (not the pixel source)")
            )
        return DecodedResource(ir, payloads)

    @staticmethod
    def _describe_dds(header: bytes) -> dict:
        info: dict = {"valid": False}
        if len(header) < DDS_HEADER_SIZE:
            info["error"] = f"header too short ({len(header)} bytes)"
            return info
        if header[:4] != DDS_MAGIC:
            info["error"] = f"bad DDS magic {header[:4]!r}"
            return info
        height, width = struct.unpack_from("<II", header, 12)
        pitch, depth, mip_count = struct.unpack_from("<III", header, 20)
        pf_flags, fourcc = struct.unpack_from("<I4s", header, 80)
        array_size = struct.unpack_from("<I", header, 140)[0]
        dxgi = struct.unpack_from("<I", header, 128)[0] if fourcc == DX10_MAGIC else None
        info.update(
            {
                "valid": True,
                "width": width,
                "height": height,
                "pitch_or_linear_size": pitch,
                "depth": depth,
                "mip_count": mip_count,
                "pixel_format_flags": pf_flags,
                "four_cc": fourcc.decode("ascii", "replace"),
                "array_size": array_size,
                "dx10": fourcc == DX10_MAGIC,
                "dxgi_format": dxgi,
                "dxgi_format_name": dxgi_format_name(dxgi) if dxgi is not None else "",
            }
        )
        return info

    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        payload = payloads.get("texture.dds")
        if payload is None:
            raise CodecError("texture: texture.dds payload is missing")
        if len(payload) < DDS_HEADER_SIZE:
            raise CodecError("texture: DDS payload too small")
        header = payload[:DDS_HEADER_SIZE]
        pixel_data = payload[DDS_HEADER_SIZE:]
        if header[:4] != DDS_MAGIC:
            raise CodecError(f"texture: not a DDS file (magic {header[:4]!r})")
        if header[84:88] != DX10_MAGIC:
            raise CodecError(
                "texture: DDS must use the DX10 extended header "
                f"(fourCC {header[84:88]!r})"
            )

        stream = Stream.writer()
        stream.uint32(int(ir.get("unknown_id", 0)))
        stream.uint32(int(ir.get("unknown1", 0)))
        stream.uint32(int(ir.get("unknown2", 0)))
        mipmaps = ir.get("mipmaps") or []
        if len(mipmaps) != 15:
            raise CodecError(f"texture: expected 15 mipmap entries, got {len(mipmaps)}")
        for item in mipmaps:
            stream.uint32(int(item.get("start", 0)))
            stream.uint32(int(item.get("bytes_left", 0)))
            stream.uint16(int(item.get("height", 0)))
            stream.uint16(int(item.get("width", 0)))
        stream.write(header)

        tail = payloads.get("toc_tail.bin", b"")
        placement = ir.get("data_placement", "gpu")
        if placement == "toc":
            stream.write(pixel_data)
            if tail:
                stream.write(tail)
            return stream.getvalue(), payloads.get("gpu.bin", b""), payloads.get("stream.bin", b"")
        if tail:
            stream.write(tail)
        toc = stream.getvalue()
        gpu = pixel_data if placement == "gpu" else payloads.get("gpu.bin", b"")
        stream_blob = pixel_data if placement == "stream" else payloads.get("stream.bin", b"")
        return toc, gpu, stream_blob

    # ------------------------------------------------------------------ #
    @staticmethod
    def digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()[:16]


codec = register(TextureCodec())
