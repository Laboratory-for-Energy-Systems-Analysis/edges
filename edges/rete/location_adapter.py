from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edges.edgelcia import EdgeLCIA


def map_aggregate_locations_clips(lcia: "EdgeLCIA"):
    """
    CLIPSpy-backed static-location classifier adapter.

    This scaffold currently reuses the python implementation to keep behavior
    identical while providing an integration point for CLIPS classification.
    """
    return lcia._map_aggregate_locations_python()


def map_dynamic_locations_clips(lcia: "EdgeLCIA"):
    """
    CLIPSpy-backed dynamic-location classifier adapter.

    This scaffold currently reuses the python implementation to keep behavior
    identical while providing an integration point for CLIPS classification.
    """
    return lcia._map_dynamic_locations_python()
