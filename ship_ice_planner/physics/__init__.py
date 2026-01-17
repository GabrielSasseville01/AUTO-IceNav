"""
Physics module for AUTO-IceNav simulation.

This module provides physics backends for ice floe simulation:
- Pymunk: 2D rigid body physics (fast, no fracturing)
- PyChrono: 3D physics with bonded DEM for realistic ice fracture

Main components:
- IceMaterialParams: Dataclass for ice material properties
- BondedAssembly: Creates bonded particle assemblies from polygon shapes
- ChronoIceSimulator: Core PyChrono-based physics simulator
- ChronoSimAdapter: Chrono compatibility layer for sim2d.py
- PymunkBackend: Pymunk wrapper with unified interface
"""

from .ice_material import IceMaterialParams
from .bonded_assembly import BondedAssembly, Bond, Particle
from .chrono_backend import ChronoIceSimulator, IceFloe
from .sim_adapter import ChronoSimAdapter
from .pymunk_backend import PymunkBackend

__all__ = [
    'IceMaterialParams',
    'BondedAssembly',
    'Bond',
    'Particle',
    'ChronoIceSimulator',
    'IceFloe',
    'ChronoSimAdapter',
    'PymunkBackend',
]
