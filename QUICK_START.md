# AUTO-IceNav Quick Start Guide

## Repository Overview

This repository implements a 2D physics simulator for ship-ice navigation using:
- **Pymunk**: Physics engine for simulating ice floes as rigid bodies
- **MSS (Marine Systems Simulator)**: Ship dynamics modeling
- **AUTO-IceNav planner**: Path planning algorithm for navigating through ice fields

## Key Components

### 1. **Ice Floe Simulation** (`ship_ice_planner/utils/sim_utils.py`)
   - Generates random ice floes with log-normal mass distribution
   - Models ice-ship collisions with realistic physics parameters
   - Applies drag forces from water on ice floes
   - Ice floes are represented as polygons with physical properties

### 2. **Ship Dynamics** (`ship_ice_planner/controller/sim_dynamics.py`)
   - Two vessel models available:
     - Full-scale PSV (6000 tons, 76.2m x 18.0m)
     - 1:45 scale PSV (90 kg, 1.84m x 0.38m)
   - 3 DoF (degrees of freedom) linear state-space model

### 3. **Simulation** (`ship_ice_planner/sim2d.py`)
   - Main simulation loop combining ship dynamics and ice physics
   - Collision detection and response
   - Real-time visualization

## Demo Scripts

### 1. **Minimal Ice Floe Simulation** (Best starting point!)
   ```bash
   python demo_pymunk_minimal_sim.py
   ```
   - Simplest demo: just ice floes and a moving ship
   - No full ship dynamics (ship moves at constant velocity)
   - Great for understanding ice floe physics
   - Press ESC to exit

### 2. **Full Ship-Ice Navigation Demo**
   ```bash
   python demo_sim2d_ship_ice_navigation.py
   ```
   - Complete simulation with ship dynamics and AUTO-IceNav planner
   - Requires experiment config file (see below)
   - Can specify ice concentration (20%, 30%, 40%, 50%)

### 3. **Ship Dynamics Only** (No ice)
   ```bash
   python demo_dynamic_positioning.py
   ```
   - Tests ship dynamics without ice physics

## Getting Started

### Step 1: Install Dependencies
```bash
cd AUTO-IceNav
pip install -r requirements.txt
```

### Step 2: Run Minimal Demo
The simplest way to see ice floes in action:
```bash
python demo_pymunk_minimal_sim.py
```

This will:
- Generate 200 random ice floes
- Create a ship moving through them
- Show real-time visualization
- Detect collisions

### Step 3: Generate Ice Field Data (for full demo)
To run the full navigation demo, you need to generate ice field configurations:
```bash
python -m ship_ice_planner.experiments.generate_rand_exp
```

This creates `data/experiment_configs.pkl` with 400 ice fields (100 for each concentration: 20%, 30%, 40%, 50%).

### Step 4: Run Full Navigation Demo
```bash
python demo_sim2d_ship_ice_navigation.py -c 0.4 -i 0
```
- `-c 0.4`: 40% ice concentration
- `-i 0`: Use ice field index 0
- Add `--no_anim` to disable visualization (much faster)

## Understanding the Code

### Ice Floe Generation
- Location: `ship_ice_planner/utils/sim_utils.py` → `generate_obstacles()`
- Creates random polygons with specified size ranges
- Mass follows log-normal distribution
- Can control: number, size range, spatial distribution

### Physics Parameters
Key parameters in `sim_utils.py`:
- `ICE_DENSITY = 900 kg/m³`
- `ICE_THICKNESS = 1.2 m`
- `ICE_SHIP_RESTITUTION = 0.1` (bounciness)
- `ICE_SHIP_FRICTION = 0.05`
- `DRAG_FORCE_CONSTANT`: Water drag on ice floes

### Simulation Loop
1. Update physics (Pymunk step)
2. Apply drag forces to ice floes
3. Update ship position (dynamics or kinematic)
4. Check collisions
5. Update visualization

## Customization Tips

### Modify Ice Floe Parameters
In `demo_pymunk_minimal_sim.py`, change:
```python
obs_params = dict(
    num_obs=200,      # Number of ice floes
    min_r=4,          # Minimum radius (m)
    max_r=20,         # Maximum radius (m)
    min_x=0,          # X bounds
    max_x=80,
    min_y=25,         # Y bounds
    max_y=200,
)
```

### Change Ship Speed
```python
ship_vel = 2  # m/s
```

### Adjust Simulation Speed
```python
dt = 0.02     # Time step (smaller = more accurate but slower)
steps = 10    # Sub-steps per iteration
```

## Next Steps for RL/Control Project

1. **Environment Interface**: The simulation can be wrapped as a Gym/RL environment
2. **State Space**: Ship pose, velocity, ice floe positions
3. **Action Space**: Control inputs (thrusters, propellers)
4. **Reward Function**: Distance to goal, collision penalties, energy consumption
5. **Observation**: Could include costmap, local ice concentration, etc.

## File Structure
```
AUTO-IceNav/
├── demo_pymunk_minimal_sim.py          # Start here!
├── demo_sim2d_ship_ice_navigation.py   # Full demo
├── ship_ice_planner/
│   ├── sim2d.py                        # Main simulation
│   ├── utils/sim_utils.py              # Ice physics
│   ├── controller/sim_dynamics.py      # Ship dynamics
│   ├── planners/                       # Path planners
│   └── experiments/                    # Experiment scripts
└── configs/                            # Configuration files
```

## Troubleshooting

- **Import errors**: Make sure you're in the AUTO-IceNav directory and dependencies are installed
- **Slow simulation**: Use `--no_anim` flag or reduce number of ice floes
- **Missing data files**: Run `generate_rand_exp.py` first

## References

- Papers: See README.md for citations
- Pymunk docs: http://www.pymunk.org/
- MSS docs: https://github.com/cybergalactic/MSS

