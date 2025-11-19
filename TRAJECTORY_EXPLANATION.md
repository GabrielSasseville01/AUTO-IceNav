# How Trajectories are Determined in AUTO-IceNav

## Overview

The trajectory planning in AUTO-IceNav uses a **two-level hierarchical approach**:

1. **Path Planning** (High-level): Uses A* search on a state lattice to find a collision-free path
2. **Trajectory Tracking** (Low-level): Controller tracks the planned path using dynamic positioning

The system operates in a **receding horizon** fashion - the planner periodically recomputes the path as the ship moves and ice conditions change.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Loop                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Planner    │─────────►│  Controller  │                 │
│  │  (A* Search) │  Path   │  (DP Control)│                 │
│  └──────────────┘          └──────────────┘                 │
│         ▲                          │                         │
│         │                          │                         │
│         │ Ship State               │ Control Inputs          │
│         │ Ice Positions            │                         │
│         │                          ▼                         │
│         │                  ┌──────────────┐                 │
│         │                  │ Ship Dynamics│                 │
│         │                  │   (MSS)      │                 │
│         └──────────────────┴──────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Process

### 1. **Path Planning (Lattice Planner)**

The planner uses **A* search on a state lattice** to find optimal paths.

#### A. Costmap Generation
- **Input**: Current ice floe positions (from physics simulation)
- **Process**: 
  - Creates a 2D costmap grid (resolution: `1/costmap.scale` meters per pixel)
  - Each grid cell has a cost based on:
    - Ice concentration (higher concentration = higher cost)
    - Collision risk (proximity to ice floes)
    - Boundary penalties
- **Output**: 2D costmap where each cell represents navigation cost

#### B. State Lattice Construction
- **Primitives**: Pre-computed motion primitives (curved paths) that the ship can follow
  - Each primitive connects two states: (x₁, y₁, θ₁) → (x₂, y₂, θ₂)
  - Primitives respect ship's turning radius constraints
  - Defined in `primitives.py` (e.g., `PRIM_8H_4` = 8 headings, 4 curvature options)
- **Swath**: For each primitive, compute the "swath" (area swept by ship footprint)
  - Used to compute collision costs along the path

#### C. A* Search
- **Start**: Current ship position and heading (x, y, θ)
- **Goal**: Target y-coordinate (goal line)
- **Search Process**:
  1. Initialize priority queue with start node
  2. Expand nodes by applying motion primitives
  3. For each neighbor node:
     - Compute cost: `g(n) = cost to reach node n`
     - Compute heuristic: `h(n) = estimated cost to goal` (uses Dubins path heuristic)
     - Priority: `f(n) = g(n) + weight × h(n)`
  4. Check if swath overlaps with ice (high cost) or boundaries
  5. Continue until goal is reached or no path exists
- **Output**: Sequence of waypoints: `[(x₁,y₁,θ₁), (x₂,y₂,θ₂), ..., (xₙ,yₙ,θₙ)]`

#### D. Path Optimization (Optional)
- If `optim: True` in config, the path is further optimized using CasADi
- Optimizes for:
  - Smoothness (curvature minimization)
  - Collision avoidance (body points on ship)
  - Path length

### 2. **Trajectory Tracking (Controller)**

The controller converts the discrete waypoint path into a smooth, time-varying trajectory.

#### A. Trajectory Setpoint Generator (`TrajectorySetpoint`)

**Purpose**: Generate smooth reference signals x_d(t), y_d(t), ψ_d(t) from discrete waypoints

**Process**:
1. **Arc Length Parameterization**:
   - Convert waypoints to arc length parameter `s` along path
   - `s_ref` advances at target speed: `s_ref += target_speed × dt`

2. **Waypoint Selection**:
   - Find waypoint closest to current `s_ref` position
   - Extract reference: `(x_ref, y_ref, ψ_ref)`

3. **Low-Pass Filtering**:
   - Smooth the reference signal to avoid abrupt changes
   - First-order filter: `x_d += dt × (x_ref - x_d) / T`
   - Time constants `T` based on ship's natural frequencies

4. **Speed Profile**:
   - Gradually accelerate to target speed
   - `current_speed += max_acceleration × dt`

**Output**: Smooth setpoints `[x_d, y_d, ψ_d]` updated every simulation step

#### B. Dynamic Positioning Controller

**Input**: Setpoints `[x_d, y_d, ψ_d]` and current state `[x, y, ψ, u, v, r]`

**Process**:
1. Compute control errors:
   - Position error: `e_x = x_d - x`, `e_y = y_d - y`
   - Heading error: `e_ψ = ψ_d - ψ`

2. Generate control inputs using MSS (Marine Systems Simulator):
   - **Thrusters**: Forward/aft tunnel thrusters
   - **Propellers**: Port/starboard main propellers
   - Uses PID-like control with ship dynamics model

**Output**: Control forces `τ = [F_x, F_y, M_z]` (surge, sway, yaw moment)

### 3. **Replanning Logic**

The system replans periodically to adapt to changing conditions:

**Trigger Conditions** (from `sim_dynamics.check_trigger_replan()`):
- Time-based: Every `track_time` seconds (default: 30s)
- Or when ship deviates significantly from path

**Replanning Process**:
1. Send current state to planner:
   - Ship position and heading
   - Updated ice floe positions (from physics simulation)
   - Ice floe masses

2. Planner computes new path (A* search)

3. Controller updates trajectory setpoint generator with new path

4. Continue tracking new path

## Key Components

### Files Involved

| Component | File | Purpose |
|-----------|------|---------|
| **Planner** | `planners/lattice.py` | Main planning loop, orchestrates A* search |
| **A* Search** | `a_star_search.py` | Core pathfinding algorithm |
| **Costmap** | `cost_map.py` | Generates navigation cost map from ice positions |
| **Primitives** | `primitives.py` | Pre-computed motion primitives |
| **Swath** | `swath.py` | Computes swept area for collision checking |
| **Setpoint Gen** | `controller/trajectory_setpoint.py` | Converts waypoints to smooth trajectory |
| **Controller** | `controller/sim_dynamics.py` | Dynamic positioning control |
| **Ship Dynamics** | `controller/supply.py` | Ship dynamics model (MSS) |

### Configuration Parameters

Key parameters in `configs/sim2d_config.yaml`:

```yaml
# Planning
planner: 'lattice'              # Planner type
horizon: 500.                   # Planning horizon (meters)
a_star:
  weight: 1                      # A* heuristic weight
costmap:
  scale: 0.5                    # Costmap resolution (1/scale m per pixel)
  collision_cost_weight: 0.00000048
prim:
  prim_name: 'PRIM_8H_4'        # Motion primitives
  scale: 15                     # Primitive scale factor

# Control
sim_dynamics:
  target_speed: 2.0             # Target speed (m/s)
  max_acceleration: 0.04        # Max acceleration (m/s²)
  track_time: 30                # Time before replanning (seconds)
```

## Example Flow

1. **t=0s**: Ship at (0, 0, π/2), goal at y=1100m
   - Planner receives initial state and ice positions
   - A* search finds path: waypoints at y=[0, 50, 100, ..., 1100]
   - Controller initializes trajectory setpoint generator

2. **t=0-30s**: Ship tracks planned path
   - Controller generates setpoints every 0.02s
   - Ship dynamics integrate control inputs
   - Physics simulation updates ice floe positions

3. **t=30s**: Replanning triggered
   - Send updated ship state and ice positions to planner
   - A* search finds new path (ice may have moved)
   - Controller updates with new path

4. **Repeat** until goal reached

## Differences from Simple Path Following

Unlike simple waypoint following, this system:

1. **Considers ship dynamics**: Path respects turning radius constraints
2. **Accounts for ice movement**: Replans as ice floes move
3. **Optimizes for collisions**: Minimizes expected collision cost, not just path length
4. **Smooth trajectories**: Low-pass filtering prevents abrupt maneuvers
5. **Real-time adaptation**: Continuously adapts to changing conditions

## For Your RL Project

If you want to replace the planner with RL:

- **State Space**: Could include costmap, ship state, nearby ice positions
- **Action Space**: Could be waypoints, or direct control inputs
- **Reward**: Distance to goal, collision penalties, energy consumption
- **Environment**: Wrap the simulation, keep the controller for smooth execution

The current controller (`TrajectorySetpoint` + `DPcontrol`) could still be used to execute RL-generated paths smoothly.

