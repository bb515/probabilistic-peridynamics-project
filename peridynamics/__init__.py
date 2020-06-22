"""
Peridynamics.

A module for defining and simulating peridynamic systems.
"""
from .model import Model
from .model import OpenCL
from .model import OpenCLProbabilistic

__all__ = [
    'Model', 'OpenCL', 'OpenCLProbabilistic'
    ]
