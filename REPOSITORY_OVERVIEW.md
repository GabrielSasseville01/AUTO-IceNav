# AUTO-IceNav Repository Overview

## What This Repository Does

This repository simulates **ship navigation through ice-covered waters**. It combines:
1. **2D Physics Simulation** (Pymunk) - Ice floes as rigid bodies that move and collide
2. **Ship Dynamics** (MSS) - Realistic ship motion with thrusters and propellers
3. **Path Planning** (AUTO-IceNav) - Algorithm to find safe paths through ice fields

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Simulation Loop                       │
│  (ship_ice_planner/sim2d.py)                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ Ship Dynamics│         │ Ice Physics  │             │
│  │   (MSS)      │◄───────►│  (Pymunk)    │             │
│  │              │ Collide │              │             │
│  └──────────────┘         └──────────────┘             │
│         │                         │                     │
│         │                         │                     │
│         ▼                         ▼                     │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   Controller │         │ Ice Floes    │             │
│  │  (Thrusters) │         │ (Polygons)   │             │
│  └──────────────┘         └──────────────┘             │
│                                                          │
│         └─────────────────┬─────────────────┘          │
│                           │                              │
│                           ▼                              │
│                  ┌──────────────┐                        │
│                  │   Planner    │                        │
│                  │ (AUTO-IceNav)│                        │
│                  └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

## Key Components Explained

### 1. Ice Floe Generation (`sim_utils.py`)

**Function**: `generate_obstacles()`
- Creates random ice floes as polygons
- Uses `scikit-image` to generate random circles, then converts to polygons
- Each floe has:
  - Random position within bounds
  - Random size (radius between min_r and max_r)
  - Mass based on area × ice density × thickness

**Example**:
```python
obs_dicts, obstacles = generate_obstacles(
    num_obs=200,    # 200 ice floes
    min_r=4,        # 4m minimum radius
    max_r=20,       # 20m maximum radius
    min_x=0, max_x=80,   # X bounds
    min_y=25, max_y=200  # Y bounds
)
```

### 2. Physics Setup (`sim_utils.py`)

**Function**: `init_pymunk_space()`
- Creates a Pymunk physics space
- Sets damping (friction) and solver iterations
- This is the "world" where physics happens

**Function**: `create_polygon()` / `generate_sim_obs()`
- Converts obstacle dictionaries into Pymunk physics bodies
- Each ice floe becomes a dynamic body that:
  - Has mass (density × area × thickness)
  - Can move and rotate
  - Collides with ship and other floes
  - Experiences drag from water

### 3. Ship Creation (`sim_utils.py`)

**Function**: `create_sim_ship()`
- Creates ship as a polygon shape in Pymunk
- Can be KINEMATIC (moved directly) or DYNAMIC (affected by forces)
- In minimal demo: KINEMATIC with constant velocity
- In full demo: DYNAMIC with forces from thrusters

### 4. Simulation Loop (`demo_pymunk_minimal_sim.py`)

The main loop does:
1. **Physics Step**: `space.step(dt)` - Advance physics by dt seconds
2. **Get State**: Read positions/velocities of all bodies
3. **Apply Forces**: Drag on ice floes from water
4. **Update Visualization**: Redraw ship and ice floes
5. **Check Collisions**: Pymunk automatically detects collisions

### 5. Collision Detection

Pymunk uses collision handlers:
- `collision_type = 1` for ship
- `collision_type = 2` for ice floes
- When they collide, callbacks are triggered:
  - `pre_solve`: Before collision response
  - `post_solve`: After collision (can log data)
  - `separate`: When objects stop touching

## How Ice Floes Move

1. **Initial State**: Ice floes start with small random velocity
2. **Collisions**: When ship hits floe, Pymunk applies impulse
3. **Drag Force**: Water drag slows down moving floes
   - Formula: `F_drag = -0.5 × ρ_water × C_d × A × v²`
   - Applied in `apply_drag_from_water()`
4. **Damping**: Pymunk space has damping (0.99 = 1% energy loss per step)

## Physics Parameters

Located in `sim_utils.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ICE_DENSITY` | 900 kg/m³ | Density of ice |
| `WATER_DENSITY` | 1025 kg/m³ | Density of seawater |
| `ICE_THICKNESS` | 1.2 m | Thickness of ice floes |
| `ICE_SHIP_RESTITUTION` | 0.1 | Bounciness (0=sticky, 1=bouncy) |
| `ICE_SHIP_FRICTION` | 0.05 | Friction coefficient |
| `CD_DRAG_COEFFICIENT` | 1.0 | Drag coefficient |

## File Structure Deep Dive

```
AUTO-IceNav/
│
├── demo_pymunk_minimal_sim.py          # ⭐ START HERE
│   └── Simplest demo: just ice + ship
│
├── demo_sim2d_ship_ice_navigation.py   # Full demo with planner
│   └── Complete system: ship dynamics + planner + ice
│
├── ship_ice_planner/
│   ├── sim2d.py                        # Main simulation orchestrator
│   │   └── Combines ship dynamics + ice physics + planner
│   │
│   ├── utils/
│   │   ├── sim_utils.py               # ⭐ Ice physics functions
│   │   │   ├── generate_obstacles()   # Create random ice floes
│   │   │   ├── init_pymunk_space()   # Setup physics world
│   │   │   ├── create_polygon()      # Create ice floe body
│   │   │   └── apply_drag_from_water() # Drag forces
│   │   │
│   │   └── plot.py                    # Visualization
│   │
│   ├── controller/
│   │   ├── sim_dynamics.py            # Ship dynamics model
│   │   │   └── SimShipDynamics class
│   │   ├── supply.py                  # Full-scale PSV model
│   │   └── NRC_supply.py              # 1:45 scale model
│   │
│   ├── planners/
│   │   ├── lattice.py                 # AUTO-IceNav planner
│   │   ├── skeleton.py                # Baseline planner
│   │   └── straight.py                # Baseline planner
│   │
│   └── experiments/
│       ├── generate_rand_exp.py       # Generate ice field datasets
│       └── sim_exp.py                 # Run batch experiments
│
└── configs/
    └── sim2d_config.yaml              # Configuration parameters
```

## Understanding the Minimal Demo

Let's walk through `demo_pymunk_minimal_sim.py`:

```python
# 1. SETUP PARAMETERS
length, width = 200, 80  # Map size in meters
ship_pose = (40, -40, π/2)  # Ship at (40, -40) facing north
ship_vel = 2  # Ship moves at 2 m/s

# 2. GENERATE ICE FLOES
obs_dicts, obstacles = generate_obstacles(
    num_obs=200, min_r=4, max_r=20, ...
)
# Returns: list of dicts with floe info, list of vertex arrays

# 3. CREATE PHYSICS WORLD
space = init_pymunk_space()  # Pymunk space with physics

# 4. ADD SHIP
ship_shape = create_sim_ship(space, vertices, pose, velocity)
# Ship is KINEMATIC (moved directly, not by forces)

# 5. ADD ICE FLOES
polygons = generate_sim_obs(space, obs_dicts)
# Each floe is DYNAMIC (moves due to forces/collisions)

# 6. SETUP COLLISION HANDLERS
handler = space.add_collision_handler(1, 2)  # Ship(1) vs Ice(2)
handler.separate = collide  # Callback when collision happens

# 7. SIMULATION LOOP
while running:
    space.step(dt)  # Advance physics
    apply_drag_from_water(...)  # Apply drag to ice
    plot.update(...)  # Update visualization
```

## For Your RL/Control Project

### Potential State Space
- Ship: position (x, y), heading (ψ), velocity (u, v, r)
- Ice: positions and velocities of nearby floes
- Goal: distance and bearing to goal

### Potential Action Space
- Thrusters: forward/aft tunnel thrusters
- Propellers: port/starboard main propellers
- Or: desired heading and speed

### Potential Reward Function
- `+reward` for making progress toward goal
- `-penalty` for collisions
- `-penalty` for energy consumption
- `+bonus` for reaching goal

### Wrapping as RL Environment
You could create a Gym environment that:
1. Wraps `sim2d.py` or `demo_pymunk_minimal_sim.py`
2. Provides `reset()` and `step(action)` methods
3. Returns observations, rewards, done flags
4. Allows controlling ship via actions

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run minimal demo**: `python demo_pymunk_minimal_sim.py`
3. **Experiment**: Modify ice floe parameters, ship speed, etc.
4. **Read code**: Start with `sim_utils.py` to understand ice physics
5. **Build on it**: Create your RL environment wrapper

## Questions to Explore

- How do ice floe collisions affect ship motion?
- What happens with different ice concentrations?
- How does the planner avoid collisions?
- Can you modify the physics parameters?
- How to add your own control algorithm?

