"""
edges: A Python package for calculating the environmental impact of products by
applying characterization factors conditioned by the context of exchanges.
"""

__all__ = ("EdgeLCIA", "CostLCIA", "get_available_methods")

__version__ = "1.0.0"

from .edgelcia import EdgeLCIA
from .costs import CostLCIA
from .utils import get_available_methods
