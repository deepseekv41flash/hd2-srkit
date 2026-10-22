"""Container formats: DSAR chunks, toc packages, bundle sets."""

from .dsar import DsarFile, DsarChunk
from .package import Package, TocEntry, TocType
from .stream_metadata import (
    POISON_STREAM_SIZE,
    ResolvedStreamSize,
    StreamDeclaration,
    StreamMetadataRepair,
    resolve_stream_size,
)
from .unit_refs import (
    list_material_refs,
    list_unit_refs,
    rewrite,
    rewrite_material,
    rewrite_unit,
)
from .empty_unit import (
    EmptyUnitAudit,
    EmptyUnitTemplate,
    audit_empty_unit,
    extract_template,
    find_empty_unit_candidates,
    pad_patch,
    sanitize_empty_unit,
)
from .bundle import BundleIndex, BundleSet, BundlePackage, BundleItem
from .repo import Repository, PackageRef

__all__ = [
    "DsarFile", "DsarChunk",
    "Package", "TocEntry", "TocType",
    "POISON_STREAM_SIZE", "ResolvedStreamSize", "StreamDeclaration",
    "StreamMetadataRepair", "resolve_stream_size",
    "rewrite", "rewrite_unit", "rewrite_material", "list_unit_refs", "list_material_refs",
    "EmptyUnitAudit", "EmptyUnitTemplate", "audit_empty_unit", "sanitize_empty_unit",
    "find_empty_unit_candidates", "extract_template", "pad_patch",
    "BundleIndex", "BundleSet", "BundlePackage", "BundleItem",
    "Repository", "PackageRef",
]
