"""
Visualization tools for PyChrono ice physics simulation.

This module provides visualization utilities for debugging, validation,
and presentation of bonded DEM ice simulations.

Main components:
- ParticleRenderer: Visualize bonded particle assemblies
- FractureVisualizer: Animate and display crack propagation
- BackendComparisonView: Side-by-side Pymunk vs Chrono comparison
- FractureAnalysisPlots: Post-simulation analysis figures
"""

from .particle_renderer import ParticleRenderer
from .fracture_visualizer import FractureVisualizer
from .comparison_view import BackendComparisonView
from .analysis_plots import FractureAnalysisPlots

__all__ = [
    'ParticleRenderer',
    'FractureVisualizer',
    'BackendComparisonView',
    'FractureAnalysisPlots',
]
