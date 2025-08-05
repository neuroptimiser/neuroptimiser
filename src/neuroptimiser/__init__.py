"""
NeurOptimiser: A neuromorphic metaheuristic framework based on spiking dynamics.

This module provides the public API to configure, run, and analyse neuromorphic optimisers.
"""

import importlib.metadata
from .solvers import NeurOptimiser

__all__ = [
    "NeurOptimiser",
]
__version__ = importlib.metadata.version("neuroptimiser")