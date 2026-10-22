"""Resource codec interface and registry.

A *codec* turns the three raw blobs of a :class:`~srkit.container.package.TocEntry`
into a JSON friendly intermediate representation (IR) plus a set of payload
files, and back again.

Payload contract
----------------
* ``editable=True``   - read back from disk when packing; edits are honoured.
* ``editable=False``  - master data (usually a raw byte range).  The file on disk
  is used verbatim, so a determined user *can* patch bytes, but the tool never
  interprets them.
* ``view=True``       - informational export (decoded vertex lists, ...).  Never
  read back; the tool deletes/regenerates these freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..container.package import TocEntry
from ..errors import UnsupportedError
from ..util.ids import NameTable, type_name

__all__ = [
    "Payload", "DecodedResource", "ResourceCodec", "get_codec", "has_codec",
    "codec_names", "decode_entry", "encode_entry",
]


@dataclass
class Payload:
    name: str
    data: bytes
    editable: bool = False
    view: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.view and self.editable:
            raise ValueError("a payload cannot be both editable and a view")


@dataclass
class DecodedResource:
    ir: dict[str, Any]
    payloads: list[Payload] = field(default_factory=list)

    def payload(self, name: str) -> Payload | None:
        for item in self.payloads:
            if item.name == name:
                return item
        return None


class ResourceCodec:
    """Base class for resource codecs."""

    #: resource type ids handled by this codec
    type_ids: tuple[int, ...] = ()
    #: human readable codec name (also stored in the IR)
    name: str = "codec"
    #: True when encode() ignores the IR and passes the master blobs through
    read_only: bool = False
    #: which of the three blobs this codec interprets.  Blobs *not* listed here
    #: are preserved verbatim by the framework (payload ``blob.<name>.bin``), so
    #: a codec can never silently drop data it does not understand - which is
    #: exactly how retail materials turned out to carry a gpu blob.
    handles: tuple[str, ...] = ("toc",)

    # -- decoding ------------------------------------------------------ #
    def decode(
        self,
        entry: TocEntry,
        names: NameTable,
        options: dict[str, Any] | None = None,
    ) -> DecodedResource:
        raise NotImplementedError

    # -- encoding ------------------------------------------------------ #
    def encode(
        self,
        ir: dict[str, Any],
        payloads: dict[str, bytes],
        entry: TocEntry,
        names: NameTable,
    ) -> tuple[bytes, bytes, bytes]:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------- #
    @staticmethod
    def base_ir(entry: TocEntry, codec_name: str) -> dict[str, Any]:
        return {
            "format": f"srkit/{codec_name}",
            "type_id": str(entry.type_id),
            "type_name": type_name(entry.type_id),
            "file_id": str(entry.file_id),
        }

    @staticmethod
    def master_payload(name: str, data: bytes, note: str = "") -> Payload:
        return Payload(name=name, data=bytes(data), editable=False, note=note)


_CODECS: dict[int, ResourceCodec] = {}
_FALLBACK: ResourceCodec | None = None


def register(codec: ResourceCodec) -> ResourceCodec:
    for type_id in codec.type_ids:
        _CODECS[int(type_id)] = codec
    return codec


def set_fallback(codec: ResourceCodec) -> ResourceCodec:
    """Codec used for resource types nobody has reverse engineered yet."""
    global _FALLBACK
    _FALLBACK = codec
    return codec


def get_codec(type_id: int) -> ResourceCodec | None:
    codec = _CODECS.get(int(type_id))
    if codec is None:
        return _FALLBACK
    return codec


def is_registered(type_id: int) -> bool:
    """True when a dedicated (non fallback) codec handles this type."""
    return int(type_id) in _CODECS


def registered_types() -> dict[int, str]:
    return {type_id: codec.name for type_id, codec in sorted(_CODECS.items())}


def has_codec(type_id: int) -> bool:
    return get_codec(type_id) is not None


def codec_names() -> list[str]:
    names = {codec.name for codec in _CODECS.values()}
    if _FALLBACK is not None:
        names.add(_FALLBACK.name)
    return sorted(names)


BLOB_NAMES = ("toc", "gpu", "stream")
_BLOB_ATTRIBUTES = {"toc": "toc_data", "gpu": "gpu_data", "stream": "stream_data"}


def preserved_blob_payload(blob_name: str, data: bytes) -> Payload:
    return Payload(
        name=f"blob.{blob_name}.bin",
        data=bytes(data),
        editable=False,
        note=f"{blob_name} blob is not interpreted by the codec, kept verbatim",
    )


def decode_entry(
    entry: TocEntry,
    names: NameTable,
    options: dict[str, Any] | None = None,
) -> DecodedResource:
    codec = get_codec(entry.type_id)
    if codec is None:
        raise UnsupportedError(
            f"no codec registered for type {type_name(entry.type_id)} "
            f"(0x{entry.type_id:016X})"
        )
    decoded = codec.decode(entry, names, options)
    # framework level safety net: keep every blob the codec does not claim
    for blob_name in BLOB_NAMES:
        if blob_name in codec.handles:
            continue
        data = getattr(entry, _BLOB_ATTRIBUTES[blob_name])
        if not data:
            continue
        if decoded.payload(f"blob.{blob_name}.bin") is None:
            decoded.payloads.append(preserved_blob_payload(blob_name, bytes(data)))
        decoded.ir.setdefault("preserved_blobs", []).append(
            {"name": blob_name, "size": len(data)}
        )
    return decoded


def codec_for_ir(ir: dict[str, Any] | None, type_id: int) -> ResourceCodec | None:
    """Pick the codec that produced ``ir``.

    A workspace unpacked with ``--raw`` stores raw blobs even for resource types
    that *do* have a codec; re-encoding must therefore follow the IR, not the
    type id, otherwise the specialized codec would be fed raw payload names it
    does not know.
    """
    if isinstance(ir, dict):
        fmt = str(ir.get("format", ""))
        if fmt == "srkit/raw" and _FALLBACK is not None:
            return _FALLBACK
    return get_codec(type_id)


def encode_entry(
    ir: dict[str, Any],
    payloads: dict[str, bytes],
    entry: TocEntry,
    names: NameTable,
) -> tuple[bytes, bytes, bytes]:
    codec = codec_for_ir(ir, entry.type_id)
    if codec is None:
        raise UnsupportedError(
            f"no codec registered for type {type_name(entry.type_id)}"
        )
    toc, gpu, stream = codec.encode(ir, payloads, entry, names)
    produced = {"toc": toc, "gpu": gpu, "stream": stream}
    # restore the blobs the codec does not claim, exactly as they were found
    for blob_name in BLOB_NAMES:
        if blob_name in codec.handles:
            continue
        preserved = payloads.get(f"blob.{blob_name}.bin")
        if preserved is None:
            preserved = bytes(getattr(entry, _BLOB_ATTRIBUTES[blob_name]))
        if preserved:
            produced[blob_name] = preserved
    return produced["toc"], produced["gpu"], produced["stream"]
