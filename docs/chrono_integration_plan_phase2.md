# PyChrono Integration Plan - Phase 2: Stabilization & Full Integration

## Current Status

**Completed (Phase 1):**
- ✅ `ChronoIceSimulator` backend with bonded DEM support
- ✅ `ChronoSimAdapter` compatibility layer
- ✅ Ice material parameters and bond mechanics
- ✅ Basic visualization tools
- ✅ Demo pipeline loading satellite-derived ice field data

**Issues to Address (Phase 2):**
1. ⚠️ Numerical instability (KE explodes to 10^160 J)
2. ⚠️ Ship-ice collision not working (ship passes through)
3. ⚠️ MPC controller integration (sim2d.py refactoring)

---

## Issue 1: Numerical Instability

### Root Cause Analysis

The kinetic energy explosion (from reasonable values to 10^160 J) indicates:

1. **Bond stiffness too high relative to time step**
   - DEM bonds with high Young's modulus (5 GPa) require very small dt
   - Current dt=0.02s is too large for stiff contact/bond dynamics

2. **Missing/incorrect damping**
   - Underdamped system causes oscillations that grow
   - Need both contact damping and bond damping

3. **Force calculation overflow**
   - Division by near-zero bond lengths
   - Uncapped forces leading to infinite accelerations

4. **Contact penetration issues**
   - Deep penetrations cause huge repulsion forces
   - Need penetration limiting or position correction

### Fixes

#### 1.1 Time Step Management
```python
# In chrono_backend.py - use adaptive or sub-stepping

def step(self, dt: float, current_time: float):
    """Step with sub-stepping for stability."""
    # Maximum stable dt for DEM is related to sqrt(m/k)
    max_stable_dt = self._compute_stable_dt()
    
    if dt > max_stable_dt:
        n_substeps = int(np.ceil(dt / max_stable_dt))
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            self._do_step(sub_dt)
    else:
        self._do_step(dt)

def _compute_stable_dt(self) -> float:
    """Compute maximum stable time step based on material properties."""
    # Critical dt ~ 0.1 * sqrt(min_particle_mass / max_stiffness)
    min_mass = self.material.density * 4/3 * np.pi * (self.particle_radius_min ** 3)
    max_stiffness = self.material.youngs_modulus
    critical_dt = 0.1 * np.sqrt(min_mass / max_stiffness)
    return min(critical_dt, 0.001)  # Cap at 1ms
```

#### 1.2 Proper Contact Material Configuration
```python
# In chrono_backend.py - _setup_contact_material()

def _setup_contact_material(self):
    """Configure contact material with proper damping."""
    mat = chrono.ChContactMaterialSMC()
    
    # Stiffness (lower than bulk modulus for stability)
    mat.SetYoungModulus(1e7)  # 10 MPa contact stiffness (not 5 GPa!)
    mat.SetPoissonRatio(0.3)
    
    # Damping - critical for stability
    mat.SetRestitution(0.1)  # Low restitution = high energy dissipation
    mat.SetFriction(0.3)
    
    # SMC-specific damping
    mat.SetKn(1e6)  # Normal contact stiffness
    mat.SetKt(1e6)  # Tangential contact stiffness
    mat.SetGn(1e4)  # Normal damping coefficient
    mat.SetGt(1e4)  # Tangential damping coefficient
    
    self._contact_material = mat
```

#### 1.3 Force Clamping
```python
# In bonded_assembly.py - Bond.compute_force()

def compute_force(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute bond force with safety limits."""
    # ... existing calculation ...
    
    # Clamp forces to physical limits
    MAX_FORCE = 1e8  # 100 MN max force
    force_mag = np.linalg.norm(force)
    if force_mag > MAX_FORCE:
        force = force * (MAX_FORCE / force_mag)
        
    return force, torque
```

#### 1.4 Position Correction for Penetration
```python
# In chrono_backend.py - handle deep penetrations

def _check_penetrations(self):
    """Correct deep penetrations to prevent force explosions."""
    MAX_PENETRATION = 0.1  # 10cm max penetration
    
    for contact in self.system.GetContactContainer().GetAllContacts():
        depth = contact.GetContactDistance()
        if abs(depth) > MAX_PENETRATION:
            # Apply position correction
            # Or flag for removal from simulation
            pass
```

### Testing for Issue 1
```python
def test_energy_conservation():
    """Verify energy remains bounded over simulation."""
    sim = ChronoIceSimulator(config)
    sim.create_ice_floe_bonded(vertices, thickness=1.0)
    
    initial_ke = sim.get_total_kinetic_energy()
    
    for _ in range(1000):
        sim.step(0.001)  # Small dt
        ke = sim.get_total_kinetic_energy()
        assert ke < initial_ke * 10, f"Energy explosion: {ke}"
        assert ke >= 0, f"Negative energy: {ke}"
        assert not np.isnan(ke), "NaN energy"
```

---

## Issue 2: Ship-Ice Collision Not Working

### Root Cause Analysis

The ship trajectory shows no deflection when encountering ice, indicating:

1. **Collision families misconfigured**
   - Ship and ice might be in families that don't collide
   
2. **Collision shapes not properly attached**
   - Body created but collision model not built

3. **Ship is kinematic instead of dynamic**
   - Kinematic bodies don't respond to contact forces

4. **Z-axis separation**
   - 3D simulation: ship and ice at different Z levels

### Fixes

#### 2.1 Ensure Collision Families Interact
```python
# In chrono_backend.py

def create_ship(self, ...):
    # ...
    # Collision family setup
    ship.GetCollisionModel().SetFamily(1)  # Ship = family 1
    # Don't disable collisions with family 2 (ice)!
    
def create_ice_floe_rigid(self, ...):
    # ...
    body.GetCollisionModel().SetFamily(2)  # Ice = family 2
    # Ensure family 2 collides with family 1
```

#### 2.2 Verify Collision Model is Built
```python
def create_ice_floe_rigid(self, vertices, thickness, ...):
    # Create body
    body = chrono.ChBody()
    
    # MUST call these in order:
    body.EnableCollision(True)
    
    # Add collision shape
    body.AddCollisionShape(
        chrono.ChCollisionShapeBox(self._contact_material, hx, hy, hz),
        chrono.ChFramed()  # Identity transform
    )
    
    # MUST build the model after adding shapes
    body.GetCollisionModel().Build()
    
    # Add to system AFTER collision is set up
    self.system.Add(body)
```

#### 2.3 Make Ship Dynamic (Not Kinematic)
```python
def create_ship(self, position, heading, mass, ...):
    # Create as dynamic body
    ship = chrono.ChBody()
    ship.SetMass(mass)
    ship.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
    ship.SetFixed(False)  # NOT fixed
    ship.SetBodyFixed(False)  # NOT kinematic
    
    # Enable collision
    ship.EnableCollision(True)
    
    # Add collision shape (box approximating ship hull)
    ship.AddCollisionShape(
        chrono.ChCollisionShapeBox(
            self._contact_material,
            length/2, beam/2, draft/2
        ),
        chrono.ChFramed()
    )
    ship.GetCollisionModel().Build()
```

#### 2.4 Ensure Same Z-Level
```python
def create_ship(self, position, ...):
    # Position at water surface (z=0)
    ship.SetPos(chrono.ChVector3d(position[0], position[1], 0))

def create_ice_floe_rigid(self, vertices, thickness, ...):
    # Ice floats at surface too
    z_position = 0  # Same as ship
    body.SetPos(chrono.ChVector3d(cx, cy, z_position))
```

#### 2.5 Debug Collision Detection
```python
def debug_collisions(self):
    """Print collision detection status."""
    print(f"Ship collision enabled: {self.ship_body.IsCollisionEnabled()}")
    print(f"Ship collision model built: {self.ship_body.GetCollisionModel().IsBuildCompleted()}")
    
    n_contacts = self.system.GetContactContainer().GetNcontacts()
    print(f"Active contacts: {n_contacts}")
    
    for floe_id, floe in self.ice_floes.items():
        if floe.body:
            print(f"Floe {floe_id} collision: {floe.body.IsCollisionEnabled()}")
```

### Testing for Issue 2
```python
def test_ship_ice_collision():
    """Verify ship collides with ice and is deflected."""
    sim = ChronoIceSimulator(config)
    
    # Create ice directly in ship's path
    ice_verts = np.array([[90, -10], [110, -10], [110, 10], [90, 10]])
    sim.create_ice_floe_rigid(ice_verts, thickness=2.0)
    
    # Create ship moving toward ice
    sim.create_ship(position=(50, 0), heading=0, initial_velocity=(5, 0))
    
    initial_ship_vel = sim.get_ship_state()['vx']
    
    # Run until collision should occur
    for _ in range(500):
        sim.step(0.01)
    
    final_ship_vel = sim.get_ship_state()['vx']
    
    # Ship should have slowed down from collision
    assert final_ship_vel < initial_ship_vel * 0.9, \
        f"Ship not deflected: initial={initial_ship_vel}, final={final_ship_vel}"
```

---

## Issue 3: MPC Controller Integration (sim2d.py Refactoring)

### Architecture Overview

Current `sim2d.py` structure:
```
┌─────────────────────────────────────────────────────────────┐
│                        sim2d.py                              │
├─────────────────────────────────────────────────────────────┤
│  1. Load config & ice data                                   │
│  2. Create Pymunk space                                      │
│  3. Create ship body (kinematic)                            │
│  4. Create ice floe bodies                                   │
│  5. Setup collision handlers                                 │
│  6. Start planner process                                    │
│  7. Main loop:                                               │
│     - Get body states (batch API)                           │
│     - Update ship state                                      │
│     - Send state to planner                                  │
│     - Receive path from planner                              │
│     - Compute control (sim_dynamics)                         │
│     - Apply forces/velocities                                │
│     - Step physics (space.step)                              │
│     - Handle fracturing                                      │
│     - Update visualization                                   │
└─────────────────────────────────────────────────────────────┘
```

### Refactoring Strategy: Backend Abstraction Layer

Create a `PhysicsBackend` interface that both Pymunk and Chrono implement:

```python
# ship_ice_planner/physics/backend_interface.py

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

class PhysicsBackend(Protocol):
    """Interface for physics simulation backends."""
    
    @abstractmethod
    def create_ship(self, vertices: np.ndarray, position: tuple, 
                   mass: float, inertia: float) -> int:
        """Create ship body, return ID."""
        ...
    
    @abstractmethod
    def create_ice_floe(self, vertices: np.ndarray, mass: float,
                       position: tuple, can_fracture: bool) -> int:
        """Create ice floe body, return ID."""
        ...
    
    @abstractmethod
    def set_ship_velocity(self, vx: float, vy: float, omega: float):
        """Set ship velocity (for kinematic control)."""
        ...
    
    @abstractmethod
    def apply_ship_force(self, fx: float, fy: float, torque: float):
        """Apply force to ship (for dynamic control)."""
        ...
    
    @abstractmethod
    def step(self, dt: float):
        """Advance simulation by dt seconds."""
        ...
    
    @abstractmethod
    def get_ship_state(self) -> dict:
        """Get ship state: {x, y, psi, vx, vy, omega}."""
        ...
    
    @abstractmethod
    def get_ice_states(self) -> list[dict]:
        """Get all ice floe states."""
        ...
    
    @abstractmethod
    def get_collision_events(self) -> list[dict]:
        """Get collision events since last call."""
        ...
    
    @abstractmethod
    def get_fracture_events(self) -> list[dict]:
        """Get fracture events since last call."""
        ...
```

### Implementation Plan

#### Phase 3.1: Create Backend Interface
```
ship_ice_planner/physics/
├── __init__.py
├── backend_interface.py      # Abstract interface
├── pymunk_backend.py         # Wrapper around existing Pymunk code
├── chrono_backend.py         # Existing Chrono implementation
└── backend_factory.py        # Factory to select backend
```

#### Phase 3.2: Wrap Existing Pymunk Code
```python
# ship_ice_planner/physics/pymunk_backend.py

class PymunkBackend(PhysicsBackend):
    """Pymunk implementation of physics backend."""
    
    def __init__(self, config: dict):
        self.space = init_pymunk_space()
        self.ship_body = None
        self.ice_bodies = {}
        self._collision_events = []
        self._setup_collision_handlers()
    
    def create_ship(self, vertices, position, mass, inertia):
        # Use existing create_sim_ship()
        shape = create_sim_ship(self.space, vertices, position, ...)
        self.ship_body = shape.body
        return 0
    
    def step(self, dt):
        self.space.step(dt)
    
    # ... implement other methods wrapping existing Pymunk calls
```

#### Phase 3.3: Update sim2d.py to Use Backend
```python
# sim2d.py - Refactored version

def sim(cfg_file: str = None, cfg: DotDict = None, ...):
    # ... config loading ...
    
    # Select physics backend based on config
    backend_type = cfg.get('physics', {}).get('backend', 'pymunk')
    
    if backend_type == 'chrono':
        from ship_ice_planner.physics.chrono_backend import ChronoIceSimulator
        from ship_ice_planner.physics.sim_adapter import ChronoSimAdapter
        physics = ChronoSimAdapter(cfg)
    else:
        from ship_ice_planner.physics.pymunk_backend import PymunkBackend
        physics = PymunkBackend(cfg)
    
    # Load ice field
    physics.load_ice_field(obs_dicts)
    
    # Create ship
    physics.create_ship(cfg.ship.vertices, start_pos, ...)
    
    # Main loop
    while running:
        # Get states
        ship_state = physics.get_ship_state()
        ice_states = physics.get_ice_states()
        
        # ... planner communication (unchanged) ...
        
        # Apply control
        velocity = sim_dynamics.compute_velocity(ship_state, path)
        physics.set_ship_velocity(velocity.vx, velocity.vy, velocity.omega)
        
        # Step physics
        physics.step(dt)
        
        # Handle events
        for event in physics.get_collision_events():
            # ... collision handling (unchanged) ...
        
        for event in physics.get_fracture_events():
            # ... fracture handling ...
        
        # Update visualization
        if plot:
            # Get transformed vertices for all bodies
            plot.update(ship_state, ice_states, ...)
```

#### Phase 3.4: Collision Event Mapping
```python
# Map Chrono contacts to sim2d collision event format

def get_collision_events(self) -> list[dict]:
    """Convert Chrono contacts to sim2d format."""
    events = []
    
    for contact_info in self._contact_buffer:
        events.append({
            'ship_contact_point': contact_info['point'],
            'ice_floe_idx': contact_info['floe_id'],
            'impulse': contact_info['impulse'],
            'total_ke': contact_info['kinetic_energy'],
        })
    
    self._contact_buffer.clear()
    return events
```

### Migration Path

1. **Minimal Changes First**: Add config flag to select backend
2. **Feature Parity**: Ensure Chrono backend produces same outputs
3. **Regression Testing**: Run same scenarios with both backends
4. **Gradual Adoption**: Default to Pymunk, opt-in to Chrono
5. **Full Migration**: Make Chrono default once validated

### Testing for Issue 3
```python
def test_backend_equivalence():
    """Both backends should produce similar results for same scenario."""
    config = load_config('test_config.yaml')
    
    # Run with Pymunk
    config['physics']['backend'] = 'pymunk'
    pymunk_results = run_simulation(config, duration=30.0)
    
    # Run with Chrono
    config['physics']['backend'] = 'chrono'
    chrono_results = run_simulation(config, duration=30.0)
    
    # Compare trajectories
    trajectory_diff = np.linalg.norm(
        pymunk_results['ship_trajectory'] - chrono_results['ship_trajectory'],
        axis=1
    )
    
    assert trajectory_diff.mean() < 5.0, \
        f"Trajectories differ too much: mean={trajectory_diff.mean():.2f}m"
```

---

## Implementation Timeline

### Week 1: Numerical Stability (Issue 1)
- [ ] Implement sub-stepping in chrono_backend.py
- [ ] Configure proper contact material damping
- [ ] Add force clamping
- [ ] Write energy conservation tests
- [ ] Validate KE remains bounded

### Week 2: Collision Detection (Issue 2)
- [ ] Fix collision family configuration
- [ ] Ensure collision models are built
- [ ] Make ship dynamic with proper collision shape
- [ ] Add collision debugging tools
- [ ] Verify ship-ice collision works

### Week 3: Backend Abstraction (Issue 3 - Part 1)
- [ ] Create PhysicsBackend interface
- [ ] Implement PymunkBackend wrapper
- [ ] Add backend factory
- [ ] Unit tests for both backends

### Week 4: sim2d.py Integration (Issue 3 - Part 2)
- [ ] Refactor sim2d.py to use backend interface
- [ ] Map collision events between backends
- [ ] Handle fracture events in visualization
- [ ] Integration tests with both backends

### Week 5: Validation & Polish
- [ ] Run full scenario comparisons
- [ ] Performance profiling
- [ ] Documentation
- [ ] Demo video showing fracturing

---

## Success Criteria

1. **Numerical Stability**: Kinetic energy stays bounded (< 10x initial) for 60s simulation
2. **Collision Working**: Ship trajectory deflected by ice within 10% of Pymunk behavior
3. **MPC Integration**: Full sim2d.py running with Chrono backend, producing navigation results
4. **Fracturing Visible**: Clear visualization of bonds breaking on ship-ice impact

---

## Files to Modify

| File | Changes |
|------|---------|
| `chrono_backend.py` | Time stepping, contact material, collision setup |
| `sim_adapter.py` | Collision event conversion, state queries |
| `bonded_assembly.py` | Force clamping, stability checks |
| `sim2d.py` | Backend abstraction, event handling |
| `pymunk_backend.py` | New file - wrapper for existing Pymunk |
| `backend_interface.py` | New file - abstract interface |

