"""
edges: A Python package for calculating the environmental impact of products by
applying characterization factors conditioned by the context of exchanges.
"""

__all__ = ("EdgeLCIA", "CostLCIA", "SupplyChain", "get_available_methods")

__version__ = "1.0.2"

from .logging_config import setup_package_logging
from .edgelcia import EdgeLCIA
from .costs import CostLCIA
from .utils import get_available_methods
from .supply_chain import SupplyChain
