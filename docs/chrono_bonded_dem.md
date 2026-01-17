# PyChrono Bonded DEM Ice Fracture Integration

## Overview

This document describes the integration of PyChrono as an alternative physics backend for ship-ice collision simulation with realistic ice fracture mechanics using the Bonded Discrete Element Method (DEM).

**Status**: ✅ Core implementation complete, ready for integration testing

## What Was Implemented

### 1. Physics Backend Architecture

A modular physics backend system allowing switching between:
- **Pymunk** (existing 2D physics)
- **PyChrono** (new 3D DEM physics with fracture)

```yaml
# configs/sim2d_config.yaml
physics:
  backend: chrono  # or 'pymunk'
```

### 2. Bonded DEM Ice Model

Based on [Celigueta et al. (2019)](https://doi.org/10.1007/s40571-019-00267-6) "A local bonded DEM approach for ice modeling":

- **Particles**: Ice floes represented as collections of spherical particles
- **Bonds**: Rigid distance constraints (`ChLinkDistance`) between adjacent particles
- **Fracture**: Bonds break when forces exceed Mohr-Coulomb failure criterion

### 3. Mohr-Coulomb Failure Criterion

Implements equations 18-19 from the paper:

```python
# Tensile failure
if normal_force > sigma_c:
    bond.break_mode = 'tensile'

# Shear failure (with friction under compression)
shear_limit = tau_c + friction * abs(compression_force)
if shear_force > shear_limit:
    bond.break_mode = 'shear'

# Mixed mode (tension + shear interaction)
if (normal_force/sigma_c) + (shear_force/tau_c) > 1:
    bond.break_mode = 'mixed'
```

### 4. Key Files

| File | Description |
|------|-------------|
| `ship_ice_planner/physics/chrono_backend.py` | Main PyChrono simulator class |
| `ship_ice_planner/physics/sim_adapter.py` | Compatibility adapter for sim2d.py |
| `ship_ice_planner/physics/bonded_assembly.py` | Bonded particle model and fracture logic |
| `ship_ice_planner/physics/ice_material.py` | Ice material parameters (from paper) |
| `ship_ice_planner/physics/pymunk_backend.py` | Pymunk wrapper for unified interface |

### 5. Visualization Tools

| File | Description |
|------|-------------|
| `ship_ice_planner/physics/visualization/particle_renderer.py` | Render bonded particles |
| `ship_ice_planner/physics/visualization/fracture_visualizer.py` | Animate fracture propagation |
| `ship_ice_planner/physics/visualization/comparison_view.py` | Side-by-side Pymunk/Chrono |
| `ship_ice_planner/physics/visualization/analysis_plots.py` | Fracture statistics plots |

## Installation

### Option 1: Conda (Recommended for PyChrono)

```bash
# Install miniforge if needed
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Create environment
conda create -n icenav python=3.12
conda activate icenav

# Install PyChrono
conda install -c conda-forge pychrono

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 2: Pip only (Mock mode - no real PyChrono)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The code automatically falls back to mock mode if PyChrono is not installed.

## Usage

### Basic Simulation

```python
from ship_ice_planner.physics.chrono_backend import ChronoIceSimulator
import numpy as np

# Configuration
config = {
    'physics': {'backend': 'chrono'},
    'ice_material': {
        'density': 400,              # kg/m³
        'tensile_strength': 80e3,    # Pa (80 kPa)
        'shear_strength': 60e3,      # Pa
        'friction': 0.577,           # tan(30°) - Mohr-Coulomb
    },
    'bonded_dem': {
        'bond_type': 'rigid',        # 'rigid' for brittle, 'spring' for elastic
    }
}

# Create simulator
sim = ChronoIceSimulator(config=config)

# Create bonded ice floe
vertices = np.array([[0, -12], [30, -12], [30, 12], [0, 12]])
floe_id = sim.create_ice_floe_bonded(vertices, thickness=0.5, particle_radius=1.5)

# Create ship
sim.create_ship(position=(-35, 0), heading=0, mass=150000, 
                length=22, beam=9, initial_velocity=(15, 0))

# Run simulation
for step in range(800):
    sim.step(0.005)

# Get results
assembly = sim.ice_floes[floe_id].assembly
fragments = assembly.get_fragments()
print(f"Fragments: {len(fragments)}")
```

### Fragment Polygon Extraction

```python
# Get convex hull polygons for each fragment
polygons = assembly.get_fragment_polygons()

# Use for MPC planner obstacles
for poly in polygons:
    planner.add_obstacle(poly)
```

## Configuration Parameters

### Ice Material (from Celigueta et al. 2019)

| Parameter | Paper Value | Description |
|-----------|-------------|-------------|
| `density` | 920 kg/m³ | Ice density |
| `youngs_modulus` | 1 GPa | Elastic modulus |
| `tensile_strength` | 1.6 MPa | Bond tensile limit |
| `shear_strength` | 1.0 MPa | Bond shear limit |
| `friction` | 0.577 | tan(30°) friction angle |

**Note**: For visible fracture at simulation scale, strengths may need scaling down (e.g., 50-100 kPa).

### Bond Types

| Type | Behavior | Use Case |
|------|----------|----------|
| `rigid` | Brittle, clean fracture | Realistic ice breaking |
| `spring` | Elastic, bouncy | Testing, soft materials |

## Test Results

### Fracture Behavior

With `bond_type: rigid` and appropriate parameters:
- **96/208 bonds broken** (~46%)
- **6 distinct fragments** (not individual particles)
- **Clean splits** along fracture planes
- **Ship stops** after transferring momentum

### Failure Mode Distribution

| Mode | Count | Description |
|------|-------|-------------|
| Tensile | 94 | Bonds pulled apart |
| Mixed | 2 | Combined tension + shear |

## Known Limitations

1. **2D Constrained**: Bodies are constrained to XY plane (Z=0) for 2D simulation
2. **Particle Scaling**: Real ice uses cm-scale particles; simulation uses m-scale
3. **Bond Strength Calibration**: May need tuning for specific scenarios
4. **MPC Integration**: Full sim2d.py integration pending

## Next Steps

1. **Full sim2d.py Integration**: Wire Chrono backend into main simulation loop
2. **MPC Controller Testing**: Verify path planning works with fragment obstacles
3. **Parameter Calibration**: Match simulation to real ice tank data
4. **Performance Optimization**: Profile and optimize for real-time

## References

- Celigueta, M.A. et al. (2019). "A local bonded DEM approach for ice modeling." *Computational Particle Mechanics*, 6:739-765. DOI: 10.1007/s40571-019-00267-6

## Authors

- Initial implementation: January 2026
- Based on conversation with Claude AI assistant
