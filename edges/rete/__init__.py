"""
CLIPSpy/RETE integration layer for exchange matching.

This package is intentionally backend-focused and keeps ``EdgeLCIA`` data
structures unchanged. The current integration is experimental.
"""

from .adapter import map_exchanges_clips

__all__ = ("map_exchanges_clips",)

