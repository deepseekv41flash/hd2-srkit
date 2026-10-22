"""srkit - a standalone Stingray resource packer/unpacker.

Pure python (stdlib only), no Blender dependency.
See ``docs/srkit-design.md`` for the architecture and ``docs/stingray-format.md``
for the on-disk formats.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import SrkitError, FormatError, CodecError, ValidationError, UnsupportedError
from .container.repo import Repository
from .container.package import Package, TocEntry, TocType
from .container.dsar import DsarFile
from .container.bundle import BundleIndex, BundleSet
from .resources import decode_entry, encode_entry, registered_types, Payload, DecodedResource
from .util.ids import NameTable, TYPE_NAMES, parse_resource_id, type_name
from .workspace import pack, unpack, verify

__all__ = [
    "__version__",
    # errors
    "SrkitError", "FormatError", "CodecError", "ValidationError", "UnsupportedError",
    # container
    "Repository", "Package", "TocEntry", "TocType", "DsarFile", "BundleIndex", "BundleSet",
    # resources
    "decode_entry", "encode_entry", "registered_types", "Payload", "DecodedResource",
    # helpers
    "NameTable", "TYPE_NAMES", "parse_resource_id", "type_name",
    # pipeline
    "unpack", "pack", "verify",
]
