"""Exception types used across srkit."""

from __future__ import annotations

__all__ = ["SrkitError", "FormatError", "CodecError", "ValidationError", "UnsupportedError"]


class SrkitError(Exception):
    """Base class for every error raised by srkit."""


class FormatError(SrkitError):
    """The input does not match the expected on-disk layout."""


class CodecError(SrkitError):
    """A resource could not be decoded/encoded."""


class UnsupportedError(SrkitError):
    """The resource (or container variant) is known but not supported yet."""


class ValidationError(SrkitError):
    """Round trip verification failed."""
