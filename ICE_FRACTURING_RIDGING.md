# Ice Fracturing and Ridging Implementation Guide

## Overview

This guide outlines how to implement **ice fracturing** and **ridging** mechanics in the AUTO-IceNav simulation. These features will make the ice model more realistic and add interesting dynamics for your RL/control project.

## Current Ice Model

### What Exists Now

- **Ice floes**: Rigid polygons with mass, density, friction
- **Collision detection**: Pymunk handles ship-ice and ice-ice collisions
- **Collision forces**: Computed in `compute_collision_force_on_ship()`
- **Drag forces**: Water drag on moving floes
- **Ice properties**: Density, thickness, restitution, friction

### Key Files

- `ship_ice_planner/utils/sim_utils.py` - Ice physics and collision handling
- `ship_ice_planner/sim2d.py` - Main simulation loop with collision callbacks
- `ship_ice_planner/geometry/polygon.py` - Polygon utilities

## 1. Ice Fracturing

### Concept

When ice floes experience high stress (from collisions or ship impact), they should **fracture** into smaller pieces.

### Implementation Strategy

#### A. Track Stress/Strain on Ice Floes

Add to ice floe tracking:
```python
# In sim_utils.py or new ice_fracture.py
class IceFloeState:
    def __init__(self, shape, initial_area):
        self.shape = shape
        self.initial_area = initial_area
        self.cumulative_impulse = 0.0  # Track total impulse received
        self.max_stress = 0.0
        self.fracture_threshold = self.compute_fracture_threshold(initial_area)
    
    def compute_fracture_threshold(self, area):
        # Larger floes can withstand more stress
        # Could use: threshold = base_threshold * (area / min_area)^exponent
        base_threshold = 1e6  # N·s (impulse threshold)
        return base_threshold * np.sqrt(area / (np.pi * 4**2))  # Scale with size
```

#### B. Monitor Collisions for Fracture Events

Modify collision handler in `sim2d.py`:

```python
# Add to post_solve_handler
def post_solve_handler(arbiter, space, data):
    nonlocal system_ke_loss, delta_ke_ice, total_impulse, collided_ob_idx, contact_pts
    ship_shape, ice_shape = arbiter.shapes
    
    # Existing code...
    
    # NEW: Track impulse for fracturing
    impulse_magnitude = np.linalg.norm(arbiter.total_impulse)
    ice_state = ice_floe_states[ice_shape.idx]
    ice_state.cumulative_impulse += impulse_magnitude
    ice_state.max_stress = max(ice_state.max_stress, impulse_magnitude)
    
    # Check if floe should fracture
    if ice_state.cumulative_impulse > ice_state.fracture_threshold:
        fracture_ice_floe(space, ice_shape, ice_state, ice_floe_states)
```

#### C. Fracture Implementation

```python
def fracture_ice_floe(space, ice_shape, ice_state, ice_floe_states):
    """
    Split an ice floe into smaller pieces when it fractures.
    
    Strategy:
    1. Remove original floe from space
    2. Create 2-4 smaller floes based on fracture pattern
    3. Distribute mass/velocity appropriately
    """
    # Get current state
    vertices = list_vec2d_to_numpy(ice_shape.get_vertices())
    center = ice_shape.body.position
    velocity = ice_shape.body.velocity
    angular_velocity = ice_shape.body.angular_velocity
    mass = ice_shape.mass
    
    # Remove original
    space.remove(ice_shape.body, ice_shape)
    
    # Determine fracture pattern (random or based on impact location)
    num_pieces = np.random.randint(2, 5)  # 2-4 pieces
    
    # Split polygon (simplified - you may want more sophisticated splitting)
    new_floes = split_polygon(vertices, center, num_pieces)
    
    # Create new floes
    for i, (new_vertices, new_center, area_fraction) in enumerate(new_floes):
        # Create new polygon
        new_shape = create_polygon(
            space,
            new_vertices,
            new_center.x,
            new_center.y,
            initial_velocity=velocity  # Inherit velocity
        )
        new_shape.collision_type = 2
        
        # Scale mass by area
        new_shape.density = ICE_DENSITY * ICE_THICKNESS
        # Pymunk will compute mass from density * area
        
        # Add some random velocity for separation
        separation_velocity = 0.5  # m/s
        angle = 2 * np.pi * i / num_pieces
        new_shape.body.velocity += Vec2d(
            separation_velocity * np.cos(angle),
            separation_velocity * np.sin(angle)
        )
        
        # Track new floe
        new_state = IceFloeState(new_shape, poly_area(new_vertices))
        ice_floe_states[new_shape.idx] = new_state
```

#### D. Polygon Splitting Algorithm

```python
def split_polygon(vertices, center, num_pieces):
    """
    Split a polygon into smaller pieces.
    
    Simple approach: Use Voronoi-like splitting from center
    More sophisticated: Use actual fracture mechanics
    """
    from scipy.spatial import Voronoi
    
    # Create points around center for Voronoi
    angles = np.linspace(0, 2*np.pi, num_pieces, endpoint=False)
    radii = np.random.uniform(0.3, 0.7, num_pieces) * np.mean([
        np.linalg.norm(v - center) for v in vertices
    ])
    
    split_points = np.array([
        [center.x + r * np.cos(a), center.y + r * np.sin(a)]
        for r, a in zip(radii, angles)
    ])
    
    # For each split point, find vertices in that region
    # (Simplified - you may want more sophisticated geometric splitting)
    new_floes = []
    for i, point in enumerate(split_points):
        # Find vertices closest to this split point
        # Then create sub-polygon
        # This is a simplified version - real implementation needs proper polygon clipping
        pass
    
    return new_floes
```

### Parameters to Add

```python
# In sim_utils.py
FRACTURE_ENABLED = True
FRACTURE_BASE_THRESHOLD = 1e6  # N·s (impulse threshold for small floe)
FRACTURE_SIZE_EXPONENT = 0.5  # How fracture threshold scales with size
FRACTURE_MIN_SIZE = 2.0  # m (minimum floe radius before it disappears)
FRACTURE_NUM_PIECES_MIN = 2
FRACTURE_NUM_PIECES_MAX = 4
FRACTURE_SEPARATION_VELOCITY = 0.5  # m/s
```

## 2. Ice Ridging

### Concept

When ice floes are pushed together (especially by the ship), they can form **ridges** - piles of ice that create additional resistance.

### Implementation Strategy

#### A. Detect Ridging Conditions

```python
class RidgeDetector:
    def __init__(self):
        self.ridge_zones = {}  # Track areas where ridging occurs
        self.ridge_thickness = {}  # Track ridge height/thickness
    
    def check_ridging_conditions(self, polygons, ship_position, ship_velocity):
        """
        Detect when ice floes are being compressed/pushed together.
        
        Conditions:
        1. Multiple floes in close proximity
        2. Ship is pushing ice in a direction
        3. Floes are being compressed (relative velocities converging)
        """
        # Find floes near ship's path
        ship_speed = np.linalg.norm(ship_velocity)
        if ship_speed < 0.5:  # Ship not moving fast enough
            return []
        
        ship_direction = ship_velocity / ship_speed
        
        # Check for floes in front of ship
        ridging_candidates = []
        for poly in polygons:
            floe_pos = poly.body.position
            relative_pos = floe_pos - ship_position
            distance = relative_pos.length
            
            # Check if floe is in front of ship
            if distance < 50:  # Within 50m
                forward_component = relative_pos.dot(ship_direction)
                if forward_component > 0:  # In front
                    # Check if floe is being pushed
                    relative_velocity = poly.body.velocity - ship_velocity
                    if relative_velocity.dot(ship_direction) < -0.1:  # Being pushed
                        ridging_candidates.append((poly, distance, forward_component))
        
        return ridging_candidates
```

#### B. Create Ridge Zones

```python
def create_ridge_zone(space, position, direction, width, length):
    """
    Create a ridge zone that applies additional resistance.
    
    A ridge is essentially a "soft obstacle" that:
    1. Exerts drag/resistance on the ship
    2. Can accumulate more ice over time
    3. Has variable thickness based on ice accumulation
    """
    # Create a static body for the ridge
    ridge_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ridge_body.position = position
    
    # Create shape (could be a polygon or segment)
    # For simplicity, use a rectangle
    ridge_shape = pymunk.Poly(
        ridge_body,
        [
            (-width/2, -length/2),
            (width/2, -length/2),
            (width/2, length/2),
            (-width/2, length/2)
        ]
    )
    
    # Ridge properties
    ridge_shape.friction = ICE_ICE_FRICTION * 2  # Higher friction
    ridge_shape.elasticity = 0.0  # No bounce
    ridge_shape.collision_type = 3  # Different from regular ice
    
    space.add(ridge_body, ridge_shape)
    return ridge_shape
```

#### C. Apply Ridge Resistance

Modify `compute_collision_force_on_ship()` or create new function:

```python
def compute_ridge_resistance(ship_shape, ridge_shapes, ship_velocity, dt):
    """
    Compute additional resistance from ice ridges.
    
    Ridge resistance increases with:
    1. Ridge thickness (more ice = more resistance)
    2. Ship speed (faster = more resistance)
    3. Time spent in ridge (continuous resistance)
    """
    total_resistance = [0.0, 0.0, 0.0]  # Fx, Fy, Mz
    
    ship_speed = np.linalg.norm(ship_velocity)
    if ship_speed < 0.1:
        return total_resistance
    
    ship_direction = ship_velocity / ship_speed
    
    for ridge_shape in ridge_shapes:
        # Check if ship is in contact with ridge
        # (Pymunk collision detection handles this)
        # For now, approximate based on distance
        
        # Ridge resistance formula (simplified)
        # F_ridge = C_ridge * A_contact * v^2 * thickness_factor
        C_RIDGE = 5000  # Ridge resistance coefficient (N·s²/m⁴)
        thickness_factor = 1.0  # Could vary with ridge thickness
        
        # Resistance opposes motion
        resistance_magnitude = C_RIDGE * ship_speed**2 * thickness_factor
        
        resistance_force = -ship_direction * resistance_magnitude
        
        total_resistance[0] += resistance_force[0]
        total_resistance[1] += resistance_force[1]
        # Torque could be computed based on contact location
    
    return total_resistance
```

#### D. Ridge Accumulation Over Time

```python
def update_ridge_accumulation(ridge_zones, polygons, dt):
    """
    Update ridge thickness as ice accumulates.
    
    When floes are pushed into a ridge zone, they contribute to ridge thickness.
    """
    for ridge_id, ridge_data in ridge_zones.items():
        ridge_shape = ridge_data['shape']
        ridge_position = ridge_shape.body.position
        
        # Find floes near ridge
        accumulated_mass = 0.0
        for poly in polygons:
            distance = (poly.body.position - ridge_position).length
            if distance < ridge_data['width']:
                # Floe contributes to ridge
                # Could remove floe or reduce its size
                accumulated_mass += poly.mass
        
        # Update ridge thickness
        # Thickness ~ sqrt(accumulated_mass / (density * area))
        ridge_data['thickness'] = np.sqrt(
            accumulated_mass / (ICE_DENSITY * ridge_data['area'])
        )
        
        # Update resistance based on thickness
        ridge_data['resistance_coefficient'] = (
            RIDGE_BASE_RESISTANCE * (1 + ridge_data['thickness'])
        )
```

### Parameters to Add

```python
# In sim_utils.py
RIDGING_ENABLED = True
RIDGE_DETECTION_DISTANCE = 50.0  # m (distance to check for ridging)
RIDGE_MIN_FLOES = 3  # Minimum floes needed to form ridge
RIDGE_BASE_RESISTANCE = 5000  # N·s²/m⁴
RIDGE_ACCUMULATION_RATE = 0.1  # How fast ice accumulates in ridge
RIDGE_MAX_THICKNESS = 5.0  # m (maximum ridge thickness)
```

## 3. Integration Points

### Modify `sim2d.py`

```python
# Add to sim() function
ice_floe_states = {}  # Track floe states for fracturing
ridge_detector = RidgeDetector() if RIDGING_ENABLED else None
ridge_zones = {}

# In collision handler
def post_solve_handler(arbiter, space, data):
    # ... existing code ...
    
    # Fracturing
    if FRACTURE_ENABLED:
        ice_state = ice_floe_states.get(ice_shape.idx)
        if ice_state:
            ice_state.cumulative_impulse += np.linalg.norm(arbiter.total_impulse)
            if ice_state.cumulative_impulse > ice_state.fracture_threshold:
                fracture_ice_floe(space, ice_shape, ice_state, ice_floe_states)

# In main simulation loop
if RIDGING_ENABLED:
    # Check for ridging conditions
    ridging_candidates = ridge_detector.check_ridging_conditions(
        polygons, ship_body.position, ship_body.velocity
    )
    
    # Create/update ridge zones
    if len(ridging_candidates) >= RIDGE_MIN_FLOES:
        create_or_update_ridge_zone(space, ridging_candidates, ridge_zones)
    
    # Apply ridge resistance
    ridge_resistance = compute_ridge_resistance(
        ship_shape, list(ridge_zones.values()), 
        ship_body.velocity, dt
    )
    # Add to environment forces
    tau_env[0] += ridge_resistance[0]  # Add to surge force
    tau_env[1] += ridge_resistance[1]  # Add to sway force
```

### Modify `sim_utils.py`

Add new functions and parameters for fracturing and ridging.

## 4. Implementation Steps

### Phase 1: Basic Fracturing
1. ✅ Add `IceFloeState` class to track floe stress
2. ✅ Modify collision handler to track impulses
3. ✅ Implement simple polygon splitting (2 pieces)
4. ✅ Test with high-impact collisions

### Phase 2: Enhanced Fracturing
1. ✅ Better polygon splitting algorithm
2. ✅ Multiple fracture patterns
3. ✅ Size-dependent fracture thresholds
4. ✅ Visualize fractures in plots

### Phase 3: Basic Ridging
1. ✅ Detect ridging conditions
2. ✅ Create ridge zones
3. ✅ Apply simple resistance model
4. ✅ Test with ship pushing ice

### Phase 4: Enhanced Ridging
1. ✅ Ridge accumulation over time
2. ✅ Variable ridge thickness
3. ✅ Continuous resistance (not just collisions)
4. ✅ Visualize ridges

## 5. Testing Strategy

### Fracturing Tests
- High-speed collision: Ship hits large floe → should fracture
- Multiple collisions: Same floe hit multiple times → cumulative stress
- Size distribution: Verify smaller floes after fracturing

### Ridging Tests
- Continuous pushing: Ship pushes ice in same direction → ridge forms
- Resistance increase: More time in ridge → more resistance
- Power consumption: Verify ship uses more power in ridges

## 6. Visualization

Add to plotting:
- **Fractured floes**: Different color or outline
- **Ridge zones**: Overlay on map showing ridge locations
- **Ridge thickness**: Color-coded by thickness
- **Fracture events**: Animate when floes break

## 7. Configuration

Add to `configs/sim2d_config.yaml`:

```yaml
ice_physics:
  fracturing:
    enabled: true
    base_threshold: 1e6  # N·s
    size_exponent: 0.5
    min_size: 2.0  # m
    num_pieces: [2, 4]
  
  ridging:
    enabled: true
    detection_distance: 50.0  # m
    min_floes: 3
    base_resistance: 5000  # N·s²/m⁴
    max_thickness: 5.0  # m
```

## 8. References

- **Ice fracture mechanics**: Look into brittle fracture models
- **Ice ridging**: Research on pressure ridges in sea ice
- **Polygon splitting**: Computational geometry algorithms (Voronoi, Delaunay)

## Next Steps

1. Start with **basic fracturing** (simplest to implement)
2. Test thoroughly before adding ridging
3. Consider performance impact (more floes = slower simulation)
4. Add visualization early to debug

Good luck with your implementation! This will make the simulation much more interesting for RL training.

