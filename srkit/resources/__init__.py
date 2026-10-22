"""Resource codecs.  Importing this package registers every codec."""

from . import raw, bones, texture, material, unit, lua  # noqa: F401  (registration)
from .base import (
    DecodedResource,
    Payload,
    ResourceCodec,
    codec_for_ir,
    codec_names,
    decode_entry,
    encode_entry,
    get_codec,
    is_registered,
    registered_types,
    set_fallback,
)

set_fallback(raw.raw_codec())

__all__ = [
    "DecodedResource", "Payload", "ResourceCodec",
    "codec_for_ir", "codec_names", "decode_entry", "encode_entry", "get_codec",
    "is_registered", "registered_types", "set_fallback",
]
