"""Fallback codec: keep the three blobs exactly as they are.

Every resource type without a dedicated codec is handled here, which is what
makes ``srkit unpack``/``pack`` lossless for the whole archive even for resource
kinds nobody has reverse engineered yet (physics, wwise, bik, ...).
"""

from __future__ import annotations

import hashlib

from ..container.package import TocEntry
from ..util.ids import NameTable
from .base import Payload, DecodedResource, ResourceCodec, register

__all__ = ["RawCodec", "codec"]


class RawCodec(ResourceCodec):
    name = "raw"
    read_only = True
    type_ids: tuple[int, ...] = ()
    handles = ("toc", "gpu", "stream")

    def decode(self, entry: TocEntry, names: NameTable, options=None) -> DecodedResource:
        ir = self.base_ir(entry, "raw")
        ir.update(
            {
                "read_only": True,
                "note": "no dedicated codec: blobs are preserved verbatim",
                "blobs": {
                    "toc": _blob_info(entry.toc_data),
                    "gpu": _blob_info(entry.gpu_data),
                    "stream": _blob_info(entry.stream_data),
                },
            }
        )
        payloads = [
            Payload("toc.bin", bytes(entry.toc_data), note="raw toc blob"),
            Payload("gpu.bin", bytes(entry.gpu_data), note="raw gpu blob"),
            Payload("stream.bin", bytes(entry.stream_data), note="raw stream blob"),
        ]
        return DecodedResource(ir, [item for item in payloads if item.data])

    def encode(self, ir, payloads, entry: TocEntry, names: NameTable):
        toc = payloads.get("toc.bin", bytes(entry.toc_data))
        gpu = payloads.get("gpu.bin", bytes(entry.gpu_data))
        stream = payloads.get("stream.bin", bytes(entry.stream_data))
        return toc, gpu, stream


def _blob_info(blob: bytes) -> dict:
    return {
        "size": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest()[:16],
    }


codec = register(RawCodec())


def raw_codec() -> RawCodec:
    return codec
