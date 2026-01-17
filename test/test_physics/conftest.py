"""
Pytest fixtures for physics module tests.
"""

import pytest
import numpy as np


@pytest.fixture
def square_vertices():
    """A 10x10 meter square polygon."""
    return np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ], dtype=float)


@pytest.fixture
def large_polygon():
    """A 30x30 meter square - large enough for bonded DEM."""
    return np.array([
        [0, 0],
        [30, 0],
        [30, 30],
        [0, 30]
    ], dtype=float)


@pytest.fixture
def irregular_polygon():
    """An irregular hexagon shape."""
    return np.array([
        [0, 5],
        [5, 0],
        [15, 0],
        [20, 5],
        [15, 15],
        [5, 15]
    ], dtype=float)


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return {
        'physics': {
            'backend': 'chrono',
            'use_gpu': False,
            'dt': 0.02,
        },
        'ice_material': {
            'density': 920.0,
            'youngs_modulus': 5.0e9,
            'poisson_ratio': 0.33,
            'tensile_strength': 0.5e6,
            'shear_strength': 0.4e6,
            'friction': 0.1,
            'thickness': 1.0,
        },
        'bonded_dem': {
            'particle_size_ratio': 0.08,
            'min_floe_area_for_bonding': 500.0,
        },
        'ship': {
            'mass': 10000.0,
            'length': 50.0,
            'beam': 10.0,
        },
    }


@pytest.fixture
def ice_material():
    """Default ice material parameters."""
    from ship_ice_planner.physics import IceMaterialParams
    return IceMaterialParams()


@pytest.fixture
def chrono_sim(default_config):
    """Chrono simulator instance with default config."""
    from ship_ice_planner.physics import ChronoIceSimulator, IceMaterialParams
    material = IceMaterialParams.from_config(default_config)
    sim = ChronoIceSimulator(material=material, config=default_config)
    yield sim
    sim.cleanup()


@pytest.fixture
def chrono_adapter(default_config):
    """Chrono adapter instance with default config."""
    from ship_ice_planner.physics import ChronoSimAdapter
    adapter = ChronoSimAdapter(default_config)
    yield adapter
    adapter.cleanup()
