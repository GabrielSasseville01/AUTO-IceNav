"""
Ice fracturing and ridging mechanics for AUTO-IceNav simulation.

This module implements:
1. Ice fracturing: Floes break into smaller pieces under stress
2. Ice ridging: Accumulation of ice creates resistance zones
"""
import math
from typing import List, Tuple, Dict
import numpy as np
import pymunk
from pymunk import Vec2d, Poly
from shapely.geometry import Polygon, Point, LineString

from ship_ice_planner.utils.sim_utils import (
    ICE_DENSITY, ICE_THICKNESS, create_polygon, list_vec2d_to_numpy
)
from ship_ice_planner.geometry.polygon import poly_area

# Fracturing parameters
FRACTURE_ENABLED = True
FRACTURE_BASE_THRESHOLD = 1e6  # N·s (impulse threshold for small floe)
FRACTURE_SIZE_EXPONENT = 0.5  # How fracture threshold scales with size
FRACTURE_MIN_SIZE = 2.0  # m (minimum floe radius before it disappears)
FRACTURE_NUM_PIECES_MIN = 2
FRACTURE_NUM_PIECES_MAX = 4
FRACTURE_SEPARATION_VELOCITY = 0.5  # m/s

# Ridging parameters
RIDGING_ENABLED = True
RIDGE_DETECTION_DISTANCE = 50.0  # m
RIDGE_MIN_FLOES = 3
RIDGE_BASE_RESISTANCE = 5000  # N·s²/m⁴
RIDGE_ACCUMULATION_RATE = 0.1
RIDGE_MAX_THICKNESS = 5.0  # m


class IceFloeState:
    """Tracks state of an ice floe for fracturing mechanics."""
    
    def __init__(self, shape: Poly, initial_area: float):
        self.shape = shape
        self.initial_area = initial_area
        self.cumulative_impulse = 0.0
        self.max_stress = 0.0
        self.fracture_threshold = self._compute_fracture_threshold(initial_area)
        self.has_fractured = False
    
    def _compute_fracture_threshold(self, area: float) -> float:
        """Compute fracture threshold based on floe size."""
        # Larger floes can withstand more stress
        min_area = np.pi * FRACTURE_MIN_SIZE ** 2
        area_ratio = max(area / min_area, 0.1)  # Prevent division by zero
        return FRACTURE_BASE_THRESHOLD * (area_ratio ** FRACTURE_SIZE_EXPONENT)
    
    def add_impulse(self, impulse_magnitude: float):
        """Add impulse from collision."""
        self.cumulative_impulse += impulse_magnitude
        self.max_stress = max(self.max_stress, impulse_magnitude)
    
    def should_fracture(self) -> bool:
        """Check if floe should fracture."""
        if self.has_fractured:
            return False
        return self.cumulative_impulse > self.fracture_threshold


class RidgeZone:
    """Represents an ice ridge zone that creates resistance."""
    
    def __init__(self, position: Vec2d, width: float, length: float):
        self.position = position
        self.width = width
        self.length = length
        self.area = width * length  # Area of ridge zone
        self.thickness = 0.0  # m
        self.accumulated_mass = 0.0
        self.shape = None  # Will be set when added to space
        self.resistance_coefficient = RIDGE_BASE_RESISTANCE
    
    def update_thickness(self, accumulated_mass: float):
        """Update ridge thickness based on accumulated ice mass."""
        self.accumulated_mass = accumulated_mass
        # Thickness ~ sqrt(mass / (density * area))
        area = self.width * self.length
        if area > 0:
            self.thickness = min(
                np.sqrt(accumulated_mass / (ICE_DENSITY * area)),
                RIDGE_MAX_THICKNESS
            )
            # Resistance increases with thickness
            self.resistance_coefficient = RIDGE_BASE_RESISTANCE * (1 + self.thickness)


def fracture_ice_floe(
    space: pymunk.Space,
    ice_shape: Poly,
    ice_state: IceFloeState,
    ice_floe_states: Dict[int, IceFloeState],
    next_idx: int = None
) -> Tuple[List[Poly], int]:
    """
    Split an ice floe into smaller pieces when it fractures.
    
    Args:
        space: Pymunk physics space
        ice_shape: The ice floe shape to fracture
        ice_state: State tracking for the floe
        ice_floe_states: Dictionary mapping shape.idx to IceFloeState
        next_idx: Next available index for new floes (if None, will use max existing + 1)
    
    Returns:
        Tuple of (list of new ice floe shapes, next available idx)
    """
    if not FRACTURE_ENABLED:
        return [], next_idx if next_idx is not None else 0
    
    # Get current state
    vertices = list_vec2d_to_numpy(ice_shape.get_vertices())
    center = ice_shape.body.position
    velocity = ice_shape.body.velocity
    angular_velocity = ice_shape.body.angular_velocity
    mass = ice_shape.mass
    area = ice_state.initial_area
    
    # Check minimum size
    radius = np.sqrt(area / np.pi)
    if radius < FRACTURE_MIN_SIZE:
        # Too small to fracture further, just remove
        space.remove(ice_shape.body, ice_shape)
        if hasattr(ice_shape, 'idx') and ice_shape.idx in ice_floe_states:
            del ice_floe_states[ice_shape.idx]
        return [], next_idx if next_idx is not None else 0
    
    # Remove original
    space.remove(ice_shape.body, ice_shape)
    if hasattr(ice_shape, 'idx'):
        if ice_shape.idx in ice_floe_states:
            del ice_floe_states[ice_shape.idx]
    
    # Determine next available index
    if next_idx is None:
        if len(ice_floe_states) > 0:
            next_idx = max(ice_floe_states.keys()) + 1
        else:
            next_idx = 0
    
    # Determine number of pieces
    num_pieces = np.random.randint(FRACTURE_NUM_PIECES_MIN, FRACTURE_NUM_PIECES_MAX + 1)
    
    # Use shapely for proper polygon splitting
    try:
        # Create shapely polygon
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid polygon
            if not poly.is_valid or poly.area < 0.1:
                # Polygon too small or still invalid, skip fracturing
                return [], next_idx
        
        # Create split lines from center
        angles = np.linspace(0, 2 * np.pi, num_pieces, endpoint=False)
        center_pt = Point(center.x, center.y)
        
        # Create a large bounding box for split lines
        max_dist = max(np.linalg.norm(v - np.array([center.x, center.y])) for v in vertices) * 2
        
        split_polygons = []
        for i, angle in enumerate(angles):
            next_angle = angles[(i + 1) % num_pieces]
            
            # Create wedge polygon for this sector
            # Points: center, point at angle, point at next_angle, back to center
            wedge_points = [
                (center.x, center.y),
                (center.x + max_dist * np.cos(angle), center.y + max_dist * np.sin(angle)),
                (center.x + max_dist * np.cos(next_angle), center.y + max_dist * np.sin(next_angle)),
            ]
            wedge = Polygon(wedge_points)
            
            # Intersect with original polygon
            try:
                sector = poly.intersection(wedge)
            except Exception:
                # Skip this sector if intersection fails
                continue
            
            if sector.is_empty or (hasattr(sector, 'area') and sector.area < 0.1):
                continue
            
            # Handle MultiPolygon (shouldn't happen but be safe)
            if hasattr(sector, 'geoms'):
                for geom in sector.geoms:
                    if isinstance(geom, Polygon) and geom.area > 0.1:  # Minimum area
                        split_polygons.append(geom)
            elif isinstance(sector, Polygon) and sector.area > 0.1:
                split_polygons.append(sector)
        
        # If splitting failed, create simple pieces
        if len(split_polygons) < 2:
            # Fallback: create 2 pieces using a line through center
            split_angle = np.random.uniform(0, 2 * np.pi)
            split_line = LineString([
                (center.x - max_dist * np.cos(split_angle), center.y - max_dist * np.sin(split_angle)),
                (center.x + max_dist * np.cos(split_angle), center.y + max_dist * np.sin(split_angle))
            ])
            # Use buffer to create two halves
            half1 = poly.buffer(0.01).difference(split_line.buffer(0.01))
            half2 = poly.difference(half1)
            split_polygons = [p for p in [half1, half2] if isinstance(p, Polygon) and p.area > 0.1]
    
    except Exception as e:
        # If shapely fails, use simple radial splitting
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress shapely warnings
        split_polygons = None
    
    # Create new floes from split polygons
    new_floes = []
    center_np = np.array([center.x, center.y])
    
    if split_polygons:
        for i, split_poly in enumerate(split_polygons):
            # Get vertices from shapely polygon
            coords = np.array(split_poly.exterior.coords[:-1])  # Remove duplicate last point
            
            if len(coords) < 3:
                continue
            
            # Compute new center (centroid of split piece)
            new_center = split_poly.centroid
            new_center_vec = Vec2d(new_center.x, new_center.y)
            
            # Convert to relative coordinates
            relative_vertices = (coords - np.array([new_center.x, new_center.y])).tolist()
            
            # Create new polygon
            new_shape = create_polygon(
                space,
                relative_vertices,
                new_center.x,
                new_center.y,
                initial_velocity=velocity
            )
            new_shape.collision_type = 2
            new_shape.idx = next_idx  # Set unique index
            next_idx += 1
            
            # Add separation velocity (radial from original center)
            separation_dir = (new_center_vec - center).normalized()
            new_shape.body.velocity += separation_dir * FRACTURE_SEPARATION_VELOCITY
            
            # Track new floe
            new_area = split_poly.area
            new_state = IceFloeState(new_shape, new_area)
            ice_floe_states[new_shape.idx] = new_state
            
            new_floes.append(new_shape)
    else:
        # Fallback: simple radial splitting
        angles = np.linspace(0, 2 * np.pi, num_pieces, endpoint=False)
        for i, angle in enumerate(angles):
            # Simple approach: take every nth vertex
            start_idx = int(i * len(vertices) / num_pieces)
            end_idx = int((i + 1) * len(vertices) / num_pieces)
            sector_vertices = vertices[start_idx:end_idx]
            
            if len(sector_vertices) < 3:
                continue
            
            # Use centroid of sector
            sector_poly = Polygon(sector_vertices)
            new_center = sector_poly.centroid
            relative_vertices = (sector_vertices - np.array([new_center.x, new_center.y])).tolist()
            
            new_shape = create_polygon(
                space,
                relative_vertices,
                new_center.x,
                new_center.y,
                initial_velocity=velocity
            )
            new_shape.collision_type = 2
            new_shape.idx = next_idx  # Set unique index
            next_idx += 1
            
            separation_dir = Vec2d(new_center.x - center.x, new_center.y - center.y).normalized()
            new_shape.body.velocity += separation_dir * FRACTURE_SEPARATION_VELOCITY
            
            new_area = sector_poly.area
            new_state = IceFloeState(new_shape, new_area)
            ice_floe_states[new_shape.idx] = new_state
            
            new_floes.append(new_shape)
    
    ice_state.has_fractured = True
    return new_floes, next_idx


def check_ridging_conditions(
    polygons: List[Poly],
    ship_position: Vec2d,
    ship_velocity: Vec2d,
    ship_direction: float
) -> List[Tuple[Poly, float, float]]:
    """
    Detect when ice floes are being compressed/pushed together (ridging conditions).
    
    Args:
        polygons: List of ice floe shapes
        ship_position: Current ship position
        ship_velocity: Current ship velocity vector
        ship_direction: Ship heading angle (radians)
    
    Returns:
        List of (floe, distance, forward_component) tuples for ridging candidates
    """
    if not RIDGING_ENABLED:
        return []
    
    ship_speed = ship_velocity.length
    if ship_speed < 0.5:  # Ship not moving fast enough
        return []
    
    ship_dir_vec = Vec2d(np.cos(ship_direction), np.sin(ship_direction))
    
    ridging_candidates = []
    for poly in polygons:
        floe_pos = poly.body.position
        relative_pos = floe_pos - ship_position
        distance = relative_pos.length
        
        # Check if floe is in front of ship
        if distance < RIDGE_DETECTION_DISTANCE:
            forward_component = relative_pos.dot(ship_dir_vec)
            if forward_component > 0:  # In front of ship
                # Check if floe is being pushed
                relative_velocity = poly.body.velocity - ship_velocity
                if relative_velocity.dot(ship_dir_vec) < -0.1:  # Being pushed
                    ridging_candidates.append((poly, distance, forward_component))
    
    return ridging_candidates


def create_ridge_zone(
    space: pymunk.Space,
    position: Vec2d,
    direction: float,
    width: float = 20.0,
    length: float = 30.0
) -> RidgeZone:
    """
    Create a ridge zone that applies additional resistance.
    
    Args:
        space: Pymunk physics space
        position: Center position of ridge
        direction: Direction angle (radians)
        width: Width of ridge zone (m)
        length: Length of ridge zone (m)
    
    Returns:
        RidgeZone object
    """
    # Create static body for ridge
    ridge_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ridge_body.position = position
    ridge_body.angle = direction
    
    # Create rectangular shape
    half_width = width / 2
    half_length = length / 2
    vertices = [
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length)
    ]
    
    ridge_shape = pymunk.Poly(ridge_body, vertices, radius=0.01)
    ridge_shape.friction = 0.7  # Higher friction than regular ice
    ridge_shape.elasticity = 0.0
    ridge_shape.collision_type = 3  # Different from regular ice (type 2)
    
    space.add(ridge_body, ridge_shape)
    
    # Create ridge zone object
    ridge_zone = RidgeZone(position, width, length)
    ridge_zone.shape = ridge_shape
    
    return ridge_zone


def compute_ridge_resistance(
    ship_velocity: Vec2d,
    ship_position: Vec2d,
    ridge_zones: List[RidgeZone],
    dt: float
) -> Tuple[float, float, float]:
    """
    Compute additional resistance from ice ridges.
    
    Args:
        ship_velocity: Ship velocity vector
        ship_position: Ship position vector
        ridge_zones: List of active ridge zones
        dt: Time step
    
    Returns:
        (Fx, Fy, Mz) resistance forces in world frame
    """
    if not RIDGING_ENABLED or len(ridge_zones) == 0:
        return (0.0, 0.0, 0.0)
    
    ship_speed = ship_velocity.length
    if ship_speed < 0.1:
        return (0.0, 0.0, 0.0)
    
    ship_direction = ship_velocity / ship_speed
    
    total_resistance = np.array([0.0, 0.0, 0.0])  # Fx, Fy, Mz
    
    for ridge_zone in ridge_zones:
        # Check if ship is within ridge bounds
        relative_pos = ship_position - ridge_zone.position
        # Simple rectangular check (could be improved with proper polygon intersection)
        if (abs(relative_pos.x) < ridge_zone.width / 2 and 
            abs(relative_pos.y) < ridge_zone.length / 2):
            
            # Resistance opposes motion
            resistance_magnitude = (
                ridge_zone.resistance_coefficient 
                * ship_speed ** 2 
                * (1 + ridge_zone.thickness)
            )
            
            resistance_force = -ship_direction * resistance_magnitude
            total_resistance[0] += resistance_force.x
            total_resistance[1] += resistance_force.y
            # Torque could be computed based on contact location
    
    return tuple(total_resistance)


# Example usage in collision handler:
"""
def post_solve_handler_with_fracturing(arbiter, space, data):
    ship_shape, ice_shape = arbiter.shapes
    
    # Existing collision tracking...
    
    # Track impulse for fracturing
    if FRACTURE_ENABLED and ice_shape.idx in ice_floe_states:
        impulse_magnitude = np.linalg.norm(arbiter.total_impulse)
        ice_state = ice_floe_states[ice_shape.idx]
        ice_state.add_impulse(impulse_magnitude)
        
        # Check if floe should fracture
        if ice_state.should_fracture():
            new_floes = fracture_ice_floe(space, ice_shape, ice_state, ice_floe_states)
            # Update polygon list if needed
"""

