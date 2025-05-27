"""
NeurOptimiser: A neuromorphic metaheuristic framework based on spiking dynamics.

This module provides the public API to configure, run, and analyse neuromorphic optimisers.
"""

__version__ = "1.0.0"

from .solvers import NeurOptimiser, AbstractSolver
from .utils import get_arch_matrix

__all__ = [
    "NeurOptimiser",
    "AbstractSolver",
    "get_arch_matrix",
]