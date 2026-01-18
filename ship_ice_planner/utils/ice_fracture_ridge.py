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
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi

from ship_ice_planner.utils.sim_utils import (
    ICE_DENSITY, ICE_THICKNESS, create_polygon, list_vec2d_to_numpy
)
from ship_ice_planner.geometry.polygon import poly_area

# Fracturing parameters
FRACTURE_ENABLED = True  # Enable SubZero-style stress-based fracturing
FRACTURE_BASE_THRESHOLD = 1e6  # N·s (impulse threshold for small floe)
FRACTURE_SIZE_EXPONENT = 0.5  # How fracture threshold scales with size
FRACTURE_MIN_SIZE = 2.0  # m (minimum floe radius before it disappears)
FRACTURE_NUM_PIECES_MIN = 2
FRACTURE_NUM_PIECES_MAX = 4
FRACTURE_SEPARATION_VELOCITY = 0.5  # m/s

# SubZero-style stress-based fracturing parameters
# Yield criterion parameters (from SubZero fracture.m)
PSTAR = 2.25e5           # Pa - Ice strength parameter (Hibler VP rheology)
C_CONCENTRATION = 20     # Concentration factor for yield strength
Q_MOHR = 5.2             # Mohr's cone shape parameter
SIGMA_C = 250e3          # Pa - Compressive strength for Mohr's cone
SIGMA_11 = -3.375e4      # Pa - Stress parameter for yield curve

# Fracture mechanism parameters
NUM_VORONOI_PIECES = 3   # Number of pieces per fracture (SubZero default)
STRESS_HISTORY_SIZE = 10 # Number of timesteps for stress averaging
CORNER_GRIND_PROB = 0.3  # Probability of corner grinding per eligible corner
CORNER_ANGLE_THRESHOLD = 120.0  # degrees - corners sharper than this may break
MIN_CORNER_FRAGMENT_AREA = 1.0  # m² - minimum area for corner fragments

# Ridging parameters
RIDGING_ENABLED = True
RIDGE_DETECTION_DISTANCE = 50.0  # m
RIDGE_MIN_FLOES = 3
RIDGE_BASE_RESISTANCE = 5000  # N·s²/m⁴
RIDGE_ACCUMULATION_RATE = 0.1
RIDGE_MAX_THICKNESS = 5.0  # m
RIDGE_COMPRESSION_DISTANCE = 5.0  # m - distance threshold for floe-to-floe compression
RIDGE_COMPRESSION_VELOCITY_THRESHOLD = 0.1  # m/s - relative velocity threshold for compression
RIDGE_DECAY_TIME = 30.0  # seconds - time for ridge to decay when not actively maintained
RIDGE_MIN_ACTIVITY_DISTANCE = 100.0  # m - distance from ship for ridge to be considered "active"
# Ice mechanics parameters for realistic resistance
ICE_COMPRESSIVE_STRENGTH = 2.0e6  # Pa (2 MPa) - typical ice compressive strength
ICE_RESISTANCE_SCALING = 0.02  # Scale factor: tuned to match old ~40kN range (was 0.001, now 0.02 for better match)
RIDGE_KEEL_TO_SAIL_RATIO = 4.0  # Keel depth is typically 4x sail height in real ridges
WATER_DENSITY = 1025.0  # kg/m³
MAX_RIDGE_RESISTANCE = 50.0e3  # N (50 kN) - maximum resistance per ridge (reduced from 200 kN)
MAX_TOTAL_RIDGE_RESISTANCE = 100.0e3  # N (100 kN) - maximum total resistance from all ridges combined

# Enhanced ridging parameters for more realistic physics
RIDGE_INITIAL_PACKING_DENSITY = 0.6  # Initial packing density (0-1) - loose ice
RIDGE_MAX_PACKING_DENSITY = 0.9  # Maximum packing density - consolidated ice
RIDGE_CONSOLIDATION_TIME = 60.0  # seconds - time for ridge to fully consolidate
RIDGE_COMPRESSION_FORCE_SCALE = 1000.0  # N·s/m - scale factor for compression-based accumulation
ICE_BULK_DENSITY = 917.0  # kg/m³ - density of consolidated ice (less than pure ice due to air pockets)
RIDGE_SAIL_HEIGHT_FACTOR = 0.3  # Sail height as fraction of total accumulated ice height


# ==============================================================================
# Yield Criterion Functions (adapted from SubZero fracture.m)
# ==============================================================================

def compute_mohr_cone_envelope() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mohr's Cone yield envelope vertices.
    
    Adapted from SubZero fracture.m lines 21-28:
    Uses Mohr-Coulomb failure criterion with parameters q and SigC.
    
    Returns:
        Tuple of (x_vertices, y_vertices, shapely_polygon) defining yield envelope
    """
    q = Q_MOHR
    SigC = SIGMA_C
    
    # Compute Mohr's cone vertices (from SubZero)
    Sig1 = (1/q + 1) * SigC / (1/q - q)
    Sig2 = q * Sig1 + SigC
    Sig11 = SIGMA_11
    Sig22 = q * Sig11 + SigC
    
    # Yield envelope vertices (negated as in SubZero)
    MohrX = np.array([-Sig1, -Sig11, -Sig22])
    MohrY = np.array([-Sig2, -Sig22, -Sig11])
    
    # Create shapely polygon for point-in-polygon testing
    vertices = list(zip(MohrX, MohrY))
    if len(vertices) >= 3:
        yield_polygon = Polygon(vertices)
    else:
        yield_polygon = None
    
    return MohrX, MohrY, yield_polygon


def compute_elliptical_yield_envelope(
    mean_thickness: float,
    concentration: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Polygon]:
    """
    Compute elliptical yield envelope (Hibler VP rheology).
    
    Adapted from SubZero fracture.m lines 8-18:
    P = Pstar * h * exp(-C * (1 - concentration))
    
    Args:
        mean_thickness: Mean ice thickness (m)
        concentration: Ice concentration (0-1)
    
    Returns:
        Tuple of (x_coords, y_coords, shapely_polygon) defining yield envelope
    """
    # Ice strength parameter
    P = PSTAR * mean_thickness * np.exp(-C_CONCENTRATION * (1 - concentration))
    
    # Ellipse parameters (from SubZero)
    a = P * np.sqrt(2) / 2
    b = a / 2
    
    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Rotate 45 degrees (as in SubZero)
    angle = np.pi / 4
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    
    # Translate to center at (-P/2, -P/2)
    x_final = x_rot - P / 2
    y_final = y_rot - P / 2
    
    # Create shapely polygon
    vertices = list(zip(x_final, y_final))
    yield_polygon = Polygon(vertices)
    
    return x_final, y_final, yield_polygon


def check_yield_criterion(
    sigma1: float,
    sigma2: float,
    yield_type: str = 'mohr',
    mean_thickness: float = 1.0,
    concentration: float = 1.0
) -> bool:
    """
    Check if principal stresses are inside the yield envelope.
    
    Args:
        sigma1: Maximum principal stress (Pa)
        sigma2: Minimum principal stress (Pa)
        yield_type: 'mohr' for Mohr's cone or 'elliptical' for Hibler VP
        mean_thickness: Mean ice thickness for elliptical criterion (m)
        concentration: Ice concentration for elliptical criterion (0-1)
    
    Returns:
        True if stresses are INSIDE yield envelope (no fracture),
        False if OUTSIDE (should fracture)
    """
    if yield_type == 'mohr':
        _, _, yield_polygon = compute_mohr_cone_envelope()
    else:  # elliptical
        _, _, yield_polygon = compute_elliptical_yield_envelope(
            mean_thickness, concentration
        )
    
    if yield_polygon is None:
        return True  # Default to no fracture if envelope invalid
    
    # Check if point (sigma1, sigma2) is inside yield envelope
    point = Point(sigma1, sigma2)
    return yield_polygon.contains(point)


def compute_fracture_probability(
    sigma1: float,
    sigma2: float,
    yield_type: str = 'mohr',
    mean_thickness: float = 1.0,
    concentration: float = 1.0
) -> float:
    """
    Compute probability of fracture based on distance from yield envelope.
    
    Returns higher probability for stresses further outside the envelope.
    
    Args:
        sigma1: Maximum principal stress (Pa)
        sigma2: Minimum principal stress (Pa)
        yield_type: 'mohr' or 'elliptical'
        mean_thickness: Mean ice thickness (m)
        concentration: Ice concentration (0-1)
    
    Returns:
        Fracture probability (0 to 1)
    """
    if yield_type == 'mohr':
        _, _, yield_polygon = compute_mohr_cone_envelope()
    else:
        _, _, yield_polygon = compute_elliptical_yield_envelope(
            mean_thickness, concentration
        )
    
    if yield_polygon is None:
        return 0.0
    
    point = Point(sigma1, sigma2)
    
    if yield_polygon.contains(point):
        return 0.0  # Inside envelope, no fracture
    
    # Distance from boundary (positive = outside)
    distance = point.distance(yield_polygon.boundary)
    
    # Convert distance to probability (saturates at 1.0)
    # Scale factor based on typical stress magnitudes
    scale = SIGMA_C / 10.0  # Characteristic stress scale
    probability = min(1.0, distance / scale)
    
    return probability


class IceFloeState:
    """
    Tracks state of an ice floe for stress-based fracturing mechanics.
    
    Adapted from SubZero's stress tensor approach:
    - Computes 2D stress tensor from contact forces and positions
    - Maintains time-averaged stress history
    - Uses principal stress analysis for yield criterion
    """
    
    def __init__(self, shape: Poly, initial_area: float, thickness: float = None):
        self.shape = shape
        self.initial_area = initial_area
        self.thickness = thickness if thickness is not None else ICE_THICKNESS
        
        # Legacy impulse-based tracking (kept for compatibility)
        self.cumulative_impulse = 0.0
        self.max_stress = 0.0
        self.fracture_threshold = self._compute_fracture_threshold(initial_area)
        self.has_fractured = False
        
        # SubZero-style stress tensor tracking
        self.stress_tensor = np.zeros((2, 2))  # Current stress tensor (2x2)
        self.stress_history = np.zeros((STRESS_HISTORY_SIZE, 2, 2))  # History buffer
        self.stress_history_idx = 0  # Current index in circular buffer
        self.stress_count = 0  # Number of stress updates received
        
        # Interaction tracking for stress computation
        self.interactions = []  # List of (force, contact_pos) tuples
        self.center_position = None  # Floe center position (updated each step)
        
        # Principal stresses (computed from tensor)
        self.principal_stress_1 = 0.0  # Maximum principal stress
        self.principal_stress_2 = 0.0  # Minimum principal stress
        
        # Corner tracking for corner grinding
        self.corner_contacts = []  # Vertices in contact with other floes
    
    def _compute_fracture_threshold(self, area: float) -> float:
        """Compute fracture threshold based on floe size (legacy method)."""
        min_area = np.pi * FRACTURE_MIN_SIZE ** 2
        area_ratio = max(area / min_area, 0.1)
        return FRACTURE_BASE_THRESHOLD * (area_ratio ** FRACTURE_SIZE_EXPONENT)
    
    def add_impulse(self, impulse_magnitude: float):
        """Add impulse from collision (legacy method)."""
        self.cumulative_impulse += impulse_magnitude
        self.max_stress = max(self.max_stress, impulse_magnitude)
    
    def update_stress_from_interactions(self, center_x: float, center_y: float):
        """
        Compute stress tensor from contact forces and positions.
        
        Adapted from SubZero calc_trajectory.m lines 9-29:
        Stress = 1/(2*area*h) * (F⊗r + r⊗F)
        
        where F is force vector, r is position vector from center to contact point.
        
        Args:
            center_x: X coordinate of floe center
            center_y: Y coordinate of floe center
        """
        self.center_position = (center_x, center_y)
        
        if len(self.interactions) == 0:
            # No interactions, record zero stress
            current_stress = np.zeros((2, 2))
        else:
            # Compute stress tensor from all interactions
            # Stress = 1/(2*area*h) * sum_k( (r_k ⊗ F_k) + (F_k ⊗ r_k) )
            stress_sum = np.zeros((2, 2))
            
            for force, contact_pos in self.interactions:
                # Relative position from center to contact point
                r = np.array([contact_pos[0] - center_x, contact_pos[1] - center_y])
                F = np.array([force[0], force[1]])
                
                # Outer products: r⊗F and F⊗r
                r_outer_F = np.outer(r, F)
                F_outer_r = np.outer(F, r)
                
                stress_sum += r_outer_F + F_outer_r
            
            # Normalize by 2 * area * thickness
            denominator = 2.0 * self.initial_area * self.thickness
            if denominator > 1e-10:
                current_stress = stress_sum / denominator
            else:
                current_stress = np.zeros((2, 2))
        
        # Store in history buffer (circular)
        self.stress_history[self.stress_history_idx] = current_stress
        self.stress_history_idx = (self.stress_history_idx + 1) % STRESS_HISTORY_SIZE
        self.stress_count = min(self.stress_count + 1, STRESS_HISTORY_SIZE)
        
        # Compute time-averaged stress tensor
        if self.stress_count > 0:
            self.stress_tensor = np.mean(self.stress_history[:self.stress_count], axis=0)
        
        # Compute principal stresses from eigenvalue decomposition
        self._compute_principal_stresses()
        
        # Clear interactions for next timestep
        self.interactions = []
    
    def _compute_principal_stresses(self):
        """
        Compute principal stresses from the stress tensor.
        
        Principal stresses are eigenvalues of the 2D stress tensor.
        σ1 = max eigenvalue (maximum principal stress)
        σ2 = min eigenvalue (minimum principal stress)
        """
        try:
            eigenvalues = np.linalg.eigvalsh(self.stress_tensor)
            self.principal_stress_1 = np.max(eigenvalues)
            self.principal_stress_2 = np.min(eigenvalues)
        except np.linalg.LinAlgError:
            # Fallback if eigenvalue computation fails
            self.principal_stress_1 = 0.0
            self.principal_stress_2 = 0.0
    
    def add_interaction(self, force: Tuple[float, float], contact_pos: Tuple[float, float]):
        """
        Add a contact interaction for stress computation.
        
        Args:
            force: (Fx, Fy) force vector at contact point
            contact_pos: (x, y) position of contact point
        """
        self.interactions.append((force, contact_pos))
    
    def add_corner_contact(self, vertex_idx: int, contact_pos: Tuple[float, float]):
        """
        Track a corner vertex that is in contact with another floe.
        
        Args:
            vertex_idx: Index of the vertex in contact
            contact_pos: (x, y) position of contact
        """
        self.corner_contacts.append((vertex_idx, contact_pos))
    
    def clear_corner_contacts(self):
        """Clear corner contact tracking for next timestep."""
        self.corner_contacts = []
    
    def should_fracture(self) -> bool:
        """
        Check if floe should fracture using legacy impulse-based method.
        
        Note: For stress-based fracturing, use should_fracture_stress() instead.
        """
        if self.has_fractured:
            return False
        return self.cumulative_impulse > self.fracture_threshold
    
    def should_fracture_stress(
        self,
        yield_type: str = 'mohr',
        mean_thickness: float = None,
        concentration: float = 1.0,
        min_area: float = None
    ) -> bool:
        """
        Check if floe should fracture using stress-based yield criterion.
        
        Adapted from SubZero fracture.m:
        - Computes principal stresses from stress tensor
        - Checks if principal stresses are outside yield envelope
        - Returns True if floe should fracture
        
        Args:
            yield_type: 'mohr' for Mohr's cone or 'elliptical' for Hibler VP
            mean_thickness: Mean ice thickness for elliptical criterion (default: self.thickness)
            concentration: Ice concentration for elliptical criterion (0-1)
            min_area: Minimum floe area (won't fracture if smaller than this)
        
        Returns:
            True if floe should fracture based on yield criterion
        """
        if self.has_fractured:
            return False
        
        # Check minimum size
        if min_area is not None and self.initial_area < min_area:
            return False
        
        # Need some stress history before fracturing
        if self.stress_count < 3:
            return False
        
        # Use self thickness if not provided
        if mean_thickness is None:
            mean_thickness = self.thickness
        
        # Check yield criterion
        inside_envelope = check_yield_criterion(
            self.principal_stress_1,
            self.principal_stress_2,
            yield_type=yield_type,
            mean_thickness=mean_thickness,
            concentration=concentration
        )
        
        # Fracture if OUTSIDE the yield envelope
        return not inside_envelope
    
    def get_fracture_probability(
        self,
        yield_type: str = 'mohr',
        mean_thickness: float = None,
        concentration: float = 1.0
    ) -> float:
        """
        Get probability of fracture based on stress state.
        
        Args:
            yield_type: 'mohr' or 'elliptical'
            mean_thickness: Mean ice thickness (default: self.thickness)
            concentration: Ice concentration (0-1)
        
        Returns:
            Fracture probability (0 to 1)
        """
        if self.has_fractured:
            return 0.0
        
        if mean_thickness is None:
            mean_thickness = self.thickness
        
        return compute_fracture_probability(
            self.principal_stress_1,
            self.principal_stress_2,
            yield_type=yield_type,
            mean_thickness=mean_thickness,
            concentration=concentration
        )
    
    def get_stress_state(self) -> Dict:
        """
        Get current stress state for analysis/debugging.
        
        Returns:
            Dictionary with stress tensor, principal stresses, etc.
        """
        return {
            'stress_tensor': self.stress_tensor.copy(),
            'principal_stress_1': self.principal_stress_1,
            'principal_stress_2': self.principal_stress_2,
            'stress_count': self.stress_count,
            'cumulative_impulse': self.cumulative_impulse,
        }


class RidgeZone:
    """Represents an ice ridge zone that creates resistance."""
    
    def __init__(self, position: Vec2d, width: float, length: float, creation_time: float = 0.0, contributing_floes: List[Poly] = None):
        self.position = position
        self.width = width
        self.length = length
        self.area = width * length  # Area of ridge zone
        self.thickness = 0.0  # m - sail height (above water)
        self.accumulated_mass = 0.0
        self.shape = None  # Will be set when added to space
        self.body = None  # Pymunk body for the ridge
        self.resistance_coefficient = RIDGE_BASE_RESISTANCE
        self.creation_time = creation_time  # Time when ridge was created
        self.maturity = 0.0  # Ridge maturity factor (0 to 1) - grows over time
        self.contributing_floes = contributing_floes if contributing_floes is not None else []  # Floes contributing to this ridge
        self.last_geometry_update = creation_time  # Time of last geometry update
        self.last_active_time = creation_time  # Time when ridge was last active (ship nearby or ice accumulating)
        self.is_active = True  # Whether ridge is currently active
        self.keel_depth = 0.0  # m - underwater depth of ridge (more realistic than just thickness)
        # Enhanced ridging properties
        self.packing_density = RIDGE_INITIAL_PACKING_DENSITY  # Ice packing density (0-1)
        self.total_ice_height = 0.0  # m - total height of ice (sail + keel)
        self.consolidation_factor = 0.0  # Consolidation factor (0-1) - increases over time
        self.average_compression_force = 0.0  # N - average compression force from contributing floes
    
    def update_thickness(self, accumulated_mass: float, current_time: float = None, compression_force: float = 0.0):
        """
        Update ridge thickness based on accumulated ice mass, compression forces, and time.
        
        Enhanced model accounts for:
        - Ice packing density (loose vs consolidated)
        - Compression forces between floes
        - Gradual consolidation over time
        - Realistic sail/keel geometry
        
        Args:
            accumulated_mass: Mass accumulated in ridge (kg)
            current_time: Current simulation time (s)
            compression_force: Average compression force from floe interactions (N)
        """
        self.accumulated_mass = accumulated_mass
        area = self.width * self.length
        
        if area > 0 and accumulated_mass > 0:
            # Update consolidation factor over time (ice becomes more densely packed)
            if current_time is not None:
                age = current_time - self.creation_time
                # Consolidation increases gradually over RIDGE_CONSOLIDATION_TIME seconds
                self.consolidation_factor = min(1.0, age / RIDGE_CONSOLIDATION_TIME)
            else:
                # Fallback: use accumulated mass as proxy for consolidation
                self.consolidation_factor = min(1.0, accumulated_mass / (ICE_DENSITY * area * 2.0))
            
            # Packing density increases with consolidation (loose ice → consolidated ice)
            # Also increases with compression force
            compression_factor = min(1.0, compression_force / 10000.0)  # Normalize compression force
            target_packing_density = RIDGE_INITIAL_PACKING_DENSITY + (
                (RIDGE_MAX_PACKING_DENSITY - RIDGE_INITIAL_PACKING_DENSITY) * 
                (self.consolidation_factor * 0.7 + compression_factor * 0.3)
            )
            # Gradually approach target packing density
            self.packing_density = 0.95 * self.packing_density + 0.05 * target_packing_density
            
            # Effective density accounts for packing (consolidated ice has less air)
            # Use bulk density for consolidated ice, regular density for loose ice
            effective_density = (ICE_BULK_DENSITY * self.packing_density + 
                                ICE_DENSITY * (1 - self.packing_density))
            
            # Total ice height (above + below water)
            # Use effective density based on packing
            self.total_ice_height = min(
                accumulated_mass / (effective_density * area),
                RIDGE_MAX_THICKNESS * (1 + RIDGE_KEEL_TO_SAIL_RATIO)  # Max total height
            )
            
            # Sail height (above water) - only a portion of total height
            # In real ridges, sail is typically 20-30% of total height
            self.thickness = self.total_ice_height * RIDGE_SAIL_HEIGHT_FACTOR
            self.thickness = min(self.thickness, RIDGE_MAX_THICKNESS)
            
            # Keel depth (underwater) - remaining height
            # More realistic: keel accounts for most of the ice mass
            self.keel_depth = self.total_ice_height - self.thickness
            
            # Ensure keel depth scales realistically (typically 4-5x sail height)
            # But cap it to prevent unrealistic values
            max_keel_from_sail = self.thickness * RIDGE_KEEL_TO_SAIL_RATIO
            if self.keel_depth > max_keel_from_sail:
                # If we have more ice than typical, distribute it
                self.keel_depth = max_keel_from_sail
                self.total_ice_height = self.thickness + self.keel_depth
            elif self.keel_depth < self.thickness * 2.0:  # Minimum keel:sail ratio of 2:1
                # Ensure minimum keel depth for realism
                self.keel_depth = self.thickness * 2.0
                self.total_ice_height = self.thickness + self.keel_depth
        
        # Update maturity: ridges form gradually over time
        # Maturity grows from 0 to 1 over ~10 seconds
        if current_time is not None:
            age = current_time - self.creation_time
            RIDGE_FORMATION_TIME = 10.0  # seconds for ridge to fully form
            self.maturity = min(1.0, age / RIDGE_FORMATION_TIME)
        else:
            # If no time provided, use thickness as proxy for maturity
            self.maturity = min(1.0, self.thickness / 1.0)  # Assume mature at 1m thickness
        
        # Store compression force for use in accumulation
        if compression_force > 0:
            self.average_compression_force = 0.9 * self.average_compression_force + 0.1 * compression_force
        
        # Resistance increases with both thickness and maturity
        # New ridges start with low resistance and build up gradually
        # Decay factor: ridges lose effectiveness when inactive
        decay_factor = 1.0 if self.is_active else max(0.0, 1.0 - (current_time - self.last_active_time) / RIDGE_DECAY_TIME) if current_time is not None else 1.0
        
        # Enhanced resistance: also depends on packing density (consolidated ice is stronger)
        packing_strength_factor = 0.5 + 0.5 * self.packing_density  # 0.5 to 1.0 multiplier
        self.resistance_coefficient = (RIDGE_BASE_RESISTANCE * (1 + self.thickness) * 
                                       self.maturity * decay_factor * packing_strength_factor)
    
    def update_activity(self, ship_position: Vec2d, current_time: float, polygons: List[Poly]):
        """
        Update ridge activity status based on ship proximity and ice accumulation.
        
        Args:
            ship_position: Current ship position
            current_time: Current simulation time
            polygons: All ice floe polygons (to check for nearby ice)
        """
        # Check if ship is nearby
        distance_to_ship = (self.position - ship_position).length
        ship_nearby = distance_to_ship < RIDGE_MIN_ACTIVITY_DISTANCE
        
        # Check if ice is still accumulating (floes nearby)
        # Use a larger radius to check for nearby floes (ridge width might be small initially)
        nearby_floes = 0
        check_radius = max(self.width, 30.0)  # At least 30m radius, or ridge width if larger
        for poly in polygons:
            floe_pos = poly.body.position
            distance = (floe_pos - self.position).length
            if distance < check_radius:
                nearby_floes += 1
        
        # Ridge is active if ship is nearby OR ice is accumulating
        # Be more lenient - only need 1 floe nearby (not 2) to keep ridge active
        was_active = self.is_active
        self.is_active = ship_nearby or nearby_floes >= 1
        
        # Update last active time
        if self.is_active:
            self.last_active_time = current_time
        elif was_active and not self.is_active:
            # Just became inactive, record the transition time
            pass
    
    def should_remove(self, current_time: float) -> bool:
        """
        Check if ridge should be removed (decayed too much or inactive too long).
        
        Args:
            current_time: Current simulation time
        
        Returns:
            True if ridge should be removed
        """
        if self.is_active:
            return False  # Don't remove active ridges
        
        # Remove if inactive for too long
        inactive_time = current_time - self.last_active_time
        return inactive_time > RIDGE_DECAY_TIME
    
    def update_geometry_from_floes(self, space: pymunk.Space, polygons: List[Poly], min_update_interval: float = 1.0, current_time: float = None, enable_dynamic_shape: bool = False):
        """
        Dynamically update ridge geometry based on positions of contributing floes.
        
        NOTE: Dynamic shape updates are disabled by default as they can cause physics instability.
        This function only updates the logical position/size, not the Pymunk shape.
        
        Args:
            space: Pymunk physics space
            polygons: All ice floe polygons
            min_update_interval: Minimum time between geometry updates (seconds)
            current_time: Current simulation time
            enable_dynamic_shape: If True, update Pymunk shape (can cause instability)
        """
        if current_time is None or (current_time - self.last_geometry_update) < min_update_interval:
            return  # Don't update too frequently
        
        # Find floes that are contributing to this ridge (within ridge bounds)
        contributing_positions = []
        for poly in polygons:
            floe_pos = poly.body.position
            relative_pos = floe_pos - self.position
            # Check if floe is within ridge bounds
            if (abs(relative_pos.x) < self.width / 2 + 5.0 and 
                abs(relative_pos.y) < self.length / 2 + 5.0):
                contributing_positions.append(floe_pos)
                if poly not in self.contributing_floes:
                    self.contributing_floes.append(poly)
        
        # Remove floes that are no longer nearby
        self.contributing_floes = [
            floe for floe in self.contributing_floes
            if floe in polygons  # Floe still exists
        ]
        
        # Update geometry if we have contributing floes
        if len(contributing_positions) >= 2:
            # Calculate new center (weighted by floe positions)
            new_center = Vec2d(
                sum(p.x for p in contributing_positions) / len(contributing_positions),
                sum(p.y for p in contributing_positions) / len(contributing_positions)
            )
            
            # Calculate extent of contributing floes
            positions_array = np.array([[p.x, p.y] for p in contributing_positions])
            extent_x = positions_array[:, 0].max() - positions_array[:, 0].min()
            extent_y = positions_array[:, 1].max() - positions_array[:, 1].min()
            
            # Update size (add margin, but don't shrink too much, and limit maximum size)
            # Limit extent to reasonable values (max 100m) to prevent ridges from growing too large
            extent_x = min(extent_x, 100.0)
            extent_y = min(extent_y, 100.0)
            new_width = max(extent_x + 10.0, self.width * 0.8)  # Allow some shrinkage
            new_length = max(extent_y + 10.0, self.length * 0.8)
            
            # Cap maximum ridge size to prevent visualization issues
            MAX_RIDGE_SIZE = 150.0  # meters
            new_width = min(new_width, MAX_RIDGE_SIZE)
            new_length = min(new_length, MAX_RIDGE_SIZE)
            
            # Update logical position and size (for resistance calculation)
            self.position = new_center
            self.width = new_width
            self.length = new_length
            self.area = new_width * new_length
            
            # Only update Pymunk shape if explicitly enabled (disabled by default for stability)
            if enable_dynamic_shape and self.shape is not None and self.body is not None:
                # Remove old shape
                space.remove(self.shape)
                
                # Create new shape with updated geometry
                half_width = new_width / 2
                half_length = new_length / 2
                vertices = [
                    (-half_width, -half_length),
                    (half_width, -half_length),
                    (half_width, half_length),
                    (-half_width, half_length)
                ]
                
                # Update body position
                self.body.position = new_center
                
                # Create new shape
                new_shape = pymunk.Poly(self.body, vertices, radius=0.01)
                new_shape.friction = 0.7
                new_shape.elasticity = 0.0
                new_shape.collision_type = 3
                
                space.add(new_shape)
                self.shape = new_shape
            
            self.last_geometry_update = current_time
    
    def can_merge_with(self, other: 'RidgeZone', merge_distance: float = 40.0) -> bool:
        """Check if this ridge can merge with another ridge."""
        distance = (self.position - other.position).length
        return distance < merge_distance
    
    def merge_with(self, other: 'RidgeZone', space: pymunk.Space, enable_dynamic_shape: bool = False) -> 'RidgeZone':
        """
        Merge this ridge with another ridge.
        Returns the merged ridge (this one, updated).
        
        NOTE: Dynamic shape updates are disabled by default for stability.
        """
        # Combine contributing floes (remove duplicates)
        all_floes = list(set(self.contributing_floes + other.contributing_floes))
        
        # Calculate merged center (weighted average)
        all_positions = [floe.body.position for floe in all_floes]
        merged_center = Vec2d(
            sum(p.x for p in all_positions) / len(all_positions),
            sum(p.y for p in all_positions) / len(all_positions)
        )
        
        # Calculate merged extent
        positions_array = np.array([[p.x, p.y] for p in all_positions])
        extent_x = positions_array[:, 0].max() - positions_array[:, 0].min()
        extent_y = positions_array[:, 1].max() - positions_array[:, 1].min()
        
        merged_width = max(extent_x + 15.0, max(self.width, other.width))
        merged_length = max(extent_y + 15.0, max(self.length, other.length))
        
        # Use earlier creation time
        merged_creation_time = min(self.creation_time, other.creation_time)
        
        # Update this ridge (logical properties)
        self.position = merged_center
        self.width = merged_width
        self.length = merged_length
        self.area = merged_width * merged_length
        self.contributing_floes = all_floes
        self.accumulated_mass += other.accumulated_mass
        self.creation_time = merged_creation_time
        
        # Only update Pymunk shapes if enabled (disabled by default for stability)
        if enable_dynamic_shape:
            # Remove other ridge's shape from space
            if other.shape is not None:
                space.remove(other.shape)
                if other.body is not None:
                    space.remove(other.body)
            
            # Update this ridge's shape
            if self.shape is not None and self.body is not None:
                space.remove(self.shape)
                
                half_width = merged_width / 2
                half_length = merged_length / 2
                vertices = [
                    (-half_width, -half_length),
                    (half_width, -half_length),
                    (half_width, half_length),
                    (-half_width, half_length)
                ]
                
                self.body.position = merged_center
                new_shape = pymunk.Poly(self.body, vertices, radius=0.01)
                new_shape.friction = 0.7
                new_shape.elasticity = 0.0
                new_shape.collision_type = 3
                
                space.add(new_shape)
                self.shape = new_shape
        else:
            # Just remove other ridge's shape, keep this one as-is
            if other.shape is not None:
                space.remove(other.shape)
                if other.body is not None:
                    space.remove(other.body)
        
        return self


def bounded_voronoi_tessellation(
    boundary_polygon: Polygon,
    num_seeds: int = NUM_VORONOI_PIECES,
    max_attempts: int = 100
) -> List[Polygon]:
    """
    Generate bounded Voronoi tessellation within a polygon.
    
    Adapted from SubZero's polybnd_voronoi.m and fracture_floe.m:
    - Places random seed points inside the polygon
    - Computes Voronoi diagram
    - Clips Voronoi cells to polygon boundary
    
    Args:
        boundary_polygon: Shapely Polygon defining the boundary
        num_seeds: Number of Voronoi seed points (= number of pieces)
        max_attempts: Maximum attempts to place valid seeds
    
    Returns:
        List of Shapely Polygons representing the Voronoi cells
    """
    if not boundary_polygon.is_valid:
        boundary_polygon = boundary_polygon.buffer(0)
        if not boundary_polygon.is_valid:
            return []
    
    # Get bounding box
    minx, miny, maxx, maxy = boundary_polygon.bounds
    rmax = max(maxx - minx, maxy - miny) / 2
    
    # Generate random seed points inside the polygon
    seeds = []
    attempts = 0
    while len(seeds) < num_seeds and attempts < max_attempts * num_seeds:
        # Generate random point within bounding box
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        if boundary_polygon.contains(point):
            seeds.append([x, y])
        attempts += 1
    
    if len(seeds) < 2:
        # Not enough seeds, can't tessellate
        return []
    
    seeds = np.array(seeds)
    
    # Add far-away points to bound the Voronoi diagram
    # This ensures all cells are bounded (SubZero uses bounding box approach)
    far_distance = rmax * 10
    far_points = np.array([
        [minx - far_distance, miny - far_distance],
        [maxx + far_distance, miny - far_distance],
        [maxx + far_distance, maxy + far_distance],
        [minx - far_distance, maxy + far_distance],
    ])
    all_points = np.vstack([seeds, far_points])
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        return []
    
    # Extract Voronoi cells for original seeds only (not far points)
    voronoi_cells = []
    for i in range(len(seeds)):
        region_idx = vor.point_region[i]
        if region_idx == -1:
            continue
        
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        
        # Get vertices of this Voronoi cell
        try:
            cell_vertices = [vor.vertices[j] for j in region]
            cell_poly = Polygon(cell_vertices)
            
            if not cell_poly.is_valid:
                cell_poly = cell_poly.buffer(0)
            
            # Clip to boundary polygon
            clipped = cell_poly.intersection(boundary_polygon)
            
            if clipped.is_empty:
                continue
            
            # Handle MultiPolygon result
            if isinstance(clipped, MultiPolygon):
                for geom in clipped.geoms:
                    if isinstance(geom, Polygon) and geom.area > 0.1:
                        voronoi_cells.append(geom)
            elif isinstance(clipped, Polygon) and clipped.area > 0.1:
                voronoi_cells.append(clipped)
                
        except Exception:
            continue
    
    return voronoi_cells


def fracture_ice_floe(
    space: pymunk.Space,
    ice_shape: Poly,
    ice_state: IceFloeState,
    ice_floe_states: Dict[int, IceFloeState],
    next_idx: int = None,
    use_voronoi: bool = True
) -> Tuple[List[Poly], int]:
    """
    Split an ice floe into smaller pieces when it fractures.
    
    Adapted from SubZero's fracture_floe.m:
    - Uses Voronoi tessellation to create realistic fracture patterns
    - Preserves mass and momentum
    - Creates new floe state tracking for each piece
    
    Args:
        space: Pymunk physics space
        ice_shape: The ice floe shape to fracture
        ice_state: State tracking for the floe
        ice_floe_states: Dictionary mapping shape.idx to IceFloeState
        next_idx: Next available index for new floes (if None, will use max existing + 1)
        use_voronoi: If True, use Voronoi tessellation; otherwise use simple wedge splitting
    
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
    thickness = ice_state.thickness
    
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
    
    # Create shapely polygon from vertices
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid or poly.area < 0.1:
                return [], next_idx
    except Exception:
        return [], next_idx
    
    split_polygons = []
    
    # Try Voronoi tessellation first (SubZero approach)
    if use_voronoi:
        split_polygons = bounded_voronoi_tessellation(
            poly, 
            num_seeds=NUM_VORONOI_PIECES
        )
    
    # Fallback to wedge splitting if Voronoi fails
    if len(split_polygons) < 2:
        num_pieces = np.random.randint(FRACTURE_NUM_PIECES_MIN, FRACTURE_NUM_PIECES_MAX + 1)
        angles = np.linspace(0, 2 * np.pi, num_pieces, endpoint=False)
        centroid = poly.centroid
        max_dist = max(np.linalg.norm(v - np.array([centroid.x, centroid.y])) for v in vertices) * 2
        
        for i, angle in enumerate(angles):
            next_angle = angles[(i + 1) % num_pieces]
            wedge_points = [
                (centroid.x, centroid.y),
                (centroid.x + max_dist * np.cos(angle), centroid.y + max_dist * np.sin(angle)),
                (centroid.x + max_dist * np.cos(next_angle), centroid.y + max_dist * np.sin(next_angle)),
            ]
            try:
                wedge = Polygon(wedge_points)
                sector = poly.intersection(wedge)
                if not sector.is_empty and sector.area > 0.1:
                    if isinstance(sector, MultiPolygon):
                        for geom in sector.geoms:
                            if isinstance(geom, Polygon) and geom.area > 0.1:
                                split_polygons.append(geom)
                    elif isinstance(sector, Polygon):
                        split_polygons.append(sector)
            except Exception:
                continue
    
    # Create new floes from split polygons
    new_floes = []
    original_area = poly.area
    
    for split_poly in split_polygons:
        try:
            # Get vertices from shapely polygon
            if hasattr(split_poly, 'exterior'):
                coords = np.array(split_poly.exterior.coords[:-1])
            else:
                continue
            
            if len(coords) < 3:
                continue
            
            # Compute local centroid of split piece (in parent's local frame)
            local_centroid = split_poly.centroid
            local_cx, local_cy = local_centroid.x, local_centroid.y
            
            # CRITICAL: Convert to WORLD coordinates
            # SubZero: FloeNEW.Xi = floe.Xi+Xi; FloeNEW.Yi = floe.Yi+Yi;
            world_cx = center.x + local_cx
            world_cy = center.y + local_cy
            
            # Convert vertices to be relative to new centroid (local frame for new body)
            relative_vertices = (coords - np.array([local_cx, local_cy])).tolist()
            
            # Create new polygon in pymunk at correct WORLD position
            new_shape = create_polygon(
                space,
                relative_vertices,
                world_cx,
                world_cy,
                initial_velocity=velocity  # Inherit parent velocity (SubZero behavior)
            )
            new_shape.collision_type = 2
            new_shape.idx = next_idx
            next_idx += 1
            
            # SubZero: pieces inherit parent velocity, no artificial separation
            # Only add minimal separation to prevent immediate re-collision
            new_center_vec = Vec2d(world_cx, world_cy)
            try:
                separation_dir = (new_center_vec - center).normalized()
            except ZeroDivisionError:
                separation_dir = Vec2d(np.random.uniform(-1, 1), np.random.uniform(-1, 1)).normalized()
            new_shape.body.velocity += separation_dir * FRACTURE_SEPARATION_VELOCITY
            
            # Preserve angular velocity (from SubZero)
            new_shape.body.angular_velocity = angular_velocity * 0.5
            
            # Track new floe with inherited thickness
            new_area = split_poly.area
            # SubZero keeps floe thickness constant; mass scales with area via the body's density*area.
            # Thickness is only used for stress normalization (Pa = N / m^2), so it should not shrink with area.
            new_state = IceFloeState(new_shape, new_area, thickness=thickness)
            ice_floe_states[new_shape.idx] = new_state
            
            new_floes.append(new_shape)
            
        except Exception:
            continue
    
    ice_state.has_fractured = True
    return new_floes, next_idx


# ==============================================================================
# Corner Grinding Functions (adapted from SubZero corners.m and frac_corner.m)
# ==============================================================================

def compute_polygon_angles(vertices: np.ndarray) -> np.ndarray:
    """
    Compute interior angles at each vertex of a polygon.
    
    Adapted from SubZero's polyangles function.
    
    Args:
        vertices: Nx2 array of polygon vertices
    
    Returns:
        Array of interior angles in degrees for each vertex
    """
    n = len(vertices)
    if n < 3:
        return np.array([])
    
    angles = np.zeros(n)
    
    for i in range(n):
        # Get three consecutive vertices
        p1 = vertices[(i - 1) % n]
        p2 = vertices[i]
        p3 = vertices[(i + 1) % n]
        
        # Vectors from p2 to neighbors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Compute angle using dot product
        dot = np.dot(v1, v2)
        det = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product z-component
        
        # Angle in radians
        angle_rad = np.arctan2(abs(det), dot)
        
        # Convert to degrees
        angles[i] = np.degrees(angle_rad)
    
    return angles


def identify_grindable_corners(
    vertices: np.ndarray,
    contact_indices: List[int],
    angle_threshold: float = CORNER_ANGLE_THRESHOLD
) -> List[int]:
    """
    Identify corners that are sharp enough to be ground off.
    
    Adapted from SubZero corners.m:
    - Find corners with angles less than threshold
    - Check if corners are in contact with other floes
    - Use stochastic selection
    
    Args:
        vertices: Nx2 array of polygon vertices
        contact_indices: List of vertex indices that are in contact
        angle_threshold: Maximum angle (degrees) for a corner to be grindable
    
    Returns:
        List of vertex indices that should be ground off
    """
    angles = compute_polygon_angles(vertices)
    if len(angles) == 0:
        return []
    
    # Normal angle for a regular polygon with this many vertices
    n = len(vertices)
    normal_angle = 180 - 360 / n
    
    grindable = []
    
    for i, angle in enumerate(angles):
        # Check if angle is sharp enough (SubZero: rand > angles/Anorm)
        is_sharp = np.random.random() > (angle / normal_angle)
        
        # Check if in contact
        is_in_contact = i in contact_indices
        
        # Both conditions must be true (SubZero uses logical AND)
        if is_sharp and is_in_contact:
            # Additional probability check
            if np.random.random() < CORNER_GRIND_PROB:
                grindable.append(i)
    
    return grindable


def break_corner(
    poly: Polygon,
    vertex_idx: int,
    vertices: np.ndarray
) -> Tuple[Polygon, Polygon]:
    """
    Break off a corner of a polygon.
    
    Adapted from SubZero frac_corner.m:
    - Creates a small triangular piece from the corner
    - Returns the remaining polygon and the corner fragment
    
    Args:
        poly: Original shapely Polygon
        vertex_idx: Index of vertex to break off
        vertices: Nx2 array of polygon vertices
    
    Returns:
        Tuple of (remaining_polygon, corner_fragment)
    """
    n = len(vertices)
    if n < 4:  # Need at least 4 vertices to break a corner
        return poly, None
    
    # Get the corner vertex and neighbors
    prev_idx = (vertex_idx - 1) % n
    next_idx = (vertex_idx + 1) % n
    
    p_prev = vertices[prev_idx]
    p_corner = vertices[vertex_idx]
    p_next = vertices[next_idx]
    
    # Calculate distances to neighbors
    d1 = np.linalg.norm(p_corner - p_prev)
    d2 = np.linalg.norm(p_corner - p_next)
    d = min(d1, d2)
    
    # Compute corner angle
    angles = compute_polygon_angles(vertices)
    if len(angles) == 0:
        return poly, None
    
    alpha = angles[vertex_idx]
    normal_angle = 180 - 360 / n
    
    # How far along each edge to cut (SubZero formula)
    # Smaller angle = larger cut
    cut_factor = max(0.1, min(0.4, alpha / normal_angle / 5))
    
    # New vertices on each edge
    new_p1 = p_corner + cut_factor * (p_prev - p_corner) / d1 * d if d1 > 0.01 else p_prev
    new_p2 = p_corner + cut_factor * (p_next - p_corner) / d2 * d if d2 > 0.01 else p_next
    
    # Create corner triangle
    corner_vertices = np.array([new_p1, p_corner, new_p2])
    
    try:
        corner_poly = Polygon(corner_vertices)
        if not corner_poly.is_valid or corner_poly.area < MIN_CORNER_FRAGMENT_AREA:
            return poly, None
    except Exception:
        return poly, None
    
    # Create remaining polygon by removing corner
    new_vertices = []
    for i in range(n):
        if i == vertex_idx:
            new_vertices.append(new_p1)
            new_vertices.append(new_p2)
        else:
            new_vertices.append(vertices[i])
    
    try:
        remaining_poly = Polygon(new_vertices)
        if not remaining_poly.is_valid:
            remaining_poly = remaining_poly.buffer(0)
        
        if remaining_poly.area < 0.1:
            return poly, None
            
    except Exception:
        return poly, None
    
    return remaining_poly, corner_poly


def grind_corners(
    space: pymunk.Space,
    ice_shape: Poly,
    ice_state: IceFloeState,
    ice_floe_states: Dict[int, IceFloeState],
    contact_vertex_indices: List[int],
    next_idx: int = None
) -> Tuple[Poly, List[Poly], int]:
    """
    Apply corner grinding to an ice floe.
    
    Adapted from SubZero corners.m:
    - Identifies sharp corners in contact with other floes
    - Breaks off corners stochastically
    - Creates small fragment floes from broken corners
    
    Args:
        space: Pymunk physics space
        ice_shape: The ice floe shape
        ice_state: State tracking for the floe
        ice_floe_states: Dictionary mapping shape.idx to IceFloeState
        contact_vertex_indices: Indices of vertices in contact with other floes
        next_idx: Next available index for new floes
    
    Returns:
        Tuple of (updated main shape, list of corner fragments, next idx)
    """
    vertices = list_vec2d_to_numpy(ice_shape.get_vertices())
    center = ice_shape.body.position
    velocity = ice_shape.body.velocity
    
    if len(vertices) < 4:
        return ice_shape, [], next_idx if next_idx is not None else 0
    
    # Determine next available index
    if next_idx is None:
        if len(ice_floe_states) > 0:
            next_idx = max(ice_floe_states.keys()) + 1
        else:
            next_idx = 0
    
    # Identify grindable corners
    grindable = identify_grindable_corners(vertices, contact_vertex_indices)
    
    if len(grindable) == 0:
        return ice_shape, [], next_idx
    
    # Create shapely polygon
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return ice_shape, [], next_idx
    
    # Break corners (process from highest index to lowest to preserve indices)
    fragments = []
    current_vertices = vertices.copy()
    current_poly = poly
    
    for corner_idx in sorted(grindable, reverse=True):
        # Adjust index if we've removed corners before this one
        adjusted_idx = corner_idx
        for prev_idx in sorted([i for i in grindable if i > corner_idx]):
            if prev_idx < adjusted_idx:
                adjusted_idx -= 1
        
        remaining, corner_fragment = break_corner(
            current_poly, 
            min(adjusted_idx, len(current_vertices) - 1),
            current_vertices
        )
        
        if corner_fragment is not None and corner_fragment.area >= MIN_CORNER_FRAGMENT_AREA:
            # Create pymunk body for fragment
            try:
                frag_coords = np.array(corner_fragment.exterior.coords[:-1])
                frag_center = corner_fragment.centroid
                frag_center_vec = Vec2d(frag_center.x, frag_center.y)
                
                relative_verts = (frag_coords - np.array([frag_center.x, frag_center.y])).tolist()
                
                # Convert to world coordinates
                world_center = Vec2d(
                    center.x + frag_center.x - poly.centroid.x,
                    center.y + frag_center.y - poly.centroid.y
                )
                
                frag_shape = create_polygon(
                    space,
                    relative_verts,
                    world_center.x,
                    world_center.y,
                    initial_velocity=velocity
                )
                frag_shape.collision_type = 2
                frag_shape.idx = next_idx
                next_idx += 1
                
                # Add small separation velocity
                try:
                    sep_dir = (frag_center_vec - Vec2d(poly.centroid.x, poly.centroid.y)).normalized()
                except ZeroDivisionError:
                    sep_dir = Vec2d(np.random.uniform(-1, 1), np.random.uniform(-1, 1)).normalized()
                frag_shape.body.velocity += sep_dir * FRACTURE_SEPARATION_VELOCITY * 0.5
                
                # Create state for fragment
                frag_state = IceFloeState(frag_shape, corner_fragment.area, thickness=ice_state.thickness)
                ice_floe_states[frag_shape.idx] = frag_state
                
                fragments.append(frag_shape)
                
            except Exception:
                pass
        
        if remaining is not None:
            current_poly = remaining
            if hasattr(remaining, 'exterior'):
                current_vertices = np.array(remaining.exterior.coords[:-1])
    
    # Update main floe if corners were removed
    if len(fragments) > 0 and current_poly.area > 0.1:
        # Remove old shape
        old_idx = getattr(ice_shape, 'idx', None)
        space.remove(ice_shape.body, ice_shape)
        if old_idx is not None and old_idx in ice_floe_states:
            del ice_floe_states[old_idx]
        
        # Create updated shape
        try:
            new_coords = np.array(current_poly.exterior.coords[:-1])
            new_center = current_poly.centroid
            relative_verts = (new_coords - np.array([new_center.x, new_center.y])).tolist()
            
            # World position
            world_center = Vec2d(
                center.x + new_center.x - poly.centroid.x,
                center.y + new_center.y - poly.centroid.y
            )
            
            new_shape = create_polygon(
                space,
                relative_verts,
                world_center.x,
                world_center.y,
                initial_velocity=velocity
            )
            new_shape.collision_type = 2
            new_shape.idx = next_idx
            next_idx += 1
            
            # Create updated state
            new_state = IceFloeState(new_shape, current_poly.area, thickness=ice_state.thickness)
            ice_floe_states[new_shape.idx] = new_state
            
            return new_shape, fragments, next_idx
            
        except Exception:
            # Failed to create new shape, return original
            return ice_shape, [], next_idx
    
    return ice_shape, fragments, next_idx


def compute_compression_force(floe_group: List[Poly]) -> float:
    """
    Compute average compression force between floes in a compressed group.
    
    Compression force is estimated from:
    - Relative velocities (converging floes)
    - Floe masses
    - Distance between floes
    
    Args:
        floe_group: List of ice floe polygons in compressed group
    
    Returns:
        Average compression force (N)
    """
    if len(floe_group) < 2:
        return 0.0
    
    total_force = 0.0
    pair_count = 0
    
    for i, poly1 in enumerate(floe_group):
        for poly2 in floe_group[i+1:]:
            # Distance between floes
            pos1 = poly1.body.position
            pos2 = poly2.body.position
            distance = (pos1 - pos2).length
            
            if distance < RIDGE_COMPRESSION_DISTANCE and distance > 0.01:
                # Relative velocity
                vel1 = poly1.body.velocity
                vel2 = poly2.body.velocity
                relative_vel = vel1 - vel2
                
                # Direction from poly1 to poly2
                direction = (pos2 - pos1)
                if direction.length > 0.01:
                    direction = direction.normalized()
                    
                    # Compression component (negative = converging)
                    compression_component = -relative_vel.dot(direction)
                    
                    if compression_component > 0:  # Converging
                        # Estimate force from relative momentum
                        # F ≈ m * dv/dt ≈ m * dv / dt
                        # Simplified: F ≈ m * v_rel / (distance / v_avg)
                        avg_mass = (poly1.mass + poly2.mass) / 2.0
                        avg_speed = (vel1.length + vel2.length) / 2.0
                        
                        # Force estimate: F = m * v² / d (simplified momentum exchange)
                        if avg_speed > 0.01 and distance > 0.01:
                            force_magnitude = avg_mass * compression_component * avg_speed / max(distance, 0.1)
                            total_force += force_magnitude
                            pair_count += 1
    
        # Return average force per pair, scaled for realism
        if pair_count > 0:
            avg_force = total_force / pair_count
            # Scale for realistic values (convert to N, then apply scaling factor)
            return avg_force * RIDGE_COMPRESSION_FORCE_SCALE  # Already in N after scaling
        return 0.0


def detect_floe_compression(
    polygons: List[Poly],
    ship_position: Vec2d,
    ship_velocity: Vec2d,
    ship_direction: float
) -> List[List[Poly]]:
    """
    Detect groups of floes that are being compressed together.
    
    This is more realistic than individual floe detection - ridges form when
    multiple floes are pushed together, not just individually pushed.
    
    Args:
        polygons: List of ice floe shapes
        ship_position: Current ship position
        ship_velocity: Current ship velocity vector
        ship_direction: Ship heading angle (radians)
    
    Returns:
        List of compressed floe groups (each group is a list of Poly objects)
    """
    if not RIDGING_ENABLED:
        return []
    
    ship_speed = ship_velocity.length
    if ship_speed < 0.5:  # Ship not moving fast enough
        return []
    
    ship_dir_vec = Vec2d(np.cos(ship_direction), np.sin(ship_direction))
    
    # Filter floes that are in front of ship and being pushed
    candidate_floes = []
    for poly in polygons:
        floe_pos = poly.body.position
        relative_pos = floe_pos - ship_position
        distance = relative_pos.length
        
        # Check if floe is in front of ship and within detection range
        if distance < RIDGE_DETECTION_DISTANCE:
            forward_component = relative_pos.dot(ship_dir_vec)
            if forward_component > 0:  # In front of ship
                # Check if floe is being pushed
                relative_velocity = poly.body.velocity - ship_velocity
                if relative_velocity.dot(ship_dir_vec) < -0.1:  # Being pushed
                    candidate_floes.append(poly)
    
    if len(candidate_floes) < 2:
        return []  # Need at least 2 floes for compression
    
    # For lower concentrations, we might need to be more lenient
    # But still require at least 2 floes to form a compressed group
    
    # Find groups of floes that are close together and converging
    compressed_groups = []
    used_floes = set()
    
    for i, poly1 in enumerate(candidate_floes):
        if poly1 in used_floes:
            continue
        
        # Find all floes compressed with this one
        compressed_group = [poly1]
        used_floes.add(poly1)
        
        for poly2 in candidate_floes[i+1:]:
            if poly2 in used_floes:
                continue
            
            # Check distance between floes
            floe1_pos = poly1.body.position
            floe2_pos = poly2.body.position
            floe_distance = (floe1_pos - floe2_pos).length
            
            if floe_distance < RIDGE_COMPRESSION_DISTANCE and floe_distance > 0.01:  # Avoid division by zero
                # Check if floes are converging (being pushed together)
                relative_vel = poly1.body.velocity - poly2.body.velocity
                # If relative velocity is small, they're moving together
                # If it's pointing from poly1 to poly2, they're converging
                to_poly2_vec = floe2_pos - floe1_pos
                if to_poly2_vec.length > 0.01:  # Avoid zero-length vector
                    to_poly2 = to_poly2_vec.normalized()
                    convergence_component = relative_vel.dot(to_poly2)
                else:
                    # Floes are very close, check relative velocity magnitude
                    convergence_component = 0.0
                
                # Negative convergence means they're getting closer
                # Small relative velocity also indicates compression
                if (convergence_component < RIDGE_COMPRESSION_VELOCITY_THRESHOLD or 
                    relative_vel.length < RIDGE_COMPRESSION_VELOCITY_THRESHOLD):
                    compressed_group.append(poly2)
                    used_floes.add(poly2)
        
        # Only add groups with at least 2 floes (compression requires multiple floes)
        if len(compressed_group) >= 2:
            compressed_groups.append(compressed_group)
    
    return compressed_groups


def check_ridging_conditions(
    polygons: List[Poly],
    ship_position: Vec2d,
    ship_velocity: Vec2d,
    ship_direction: float
) -> Tuple[List[Tuple[Poly, float, float]], List[List[Poly]]]:
    """
    Detect when ice floes are being compressed/pushed together (ridging conditions).
    
    Now uses floe-to-floe compression detection for more realistic ridging.
    
    Args:
        polygons: List of ice floe shapes
        ship_position: Current ship position
        ship_velocity: Current ship velocity vector
        ship_direction: Ship heading angle (radians)
    
    Returns:
        Tuple of:
        - List of (floe, distance, forward_component) tuples for backward compatibility
        - List of compressed floe groups (new, more realistic detection)
    """
    if not RIDGING_ENABLED:
        return [], []
    
    ship_speed = ship_velocity.length
    if ship_speed < 0.5:  # Ship not moving fast enough
        return [], []
    
    ship_dir_vec = Vec2d(np.cos(ship_direction), np.sin(ship_direction))
    
    # New: Detect floe-to-floe compression
    compressed_groups = detect_floe_compression(
        polygons, ship_position, ship_velocity, ship_direction
    )
    
    # Legacy: Individual floe detection (for backward compatibility)
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
    
    return ridging_candidates, compressed_groups


def create_ridge_zone(
    space: pymunk.Space,
    position: Vec2d,
    direction: float,
    width: float = 20.0,
    length: float = 30.0,
    creation_time: float = 0.0,
    **kwargs
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
    # contributing_floes will be set when ridge is created from compressed group
    contributing_floes = kwargs.get('contributing_floes', [])
    ridge_zone = RidgeZone(position, width, length, creation_time=creation_time, contributing_floes=contributing_floes)
    ridge_zone.shape = ridge_shape
    ridge_zone.body = ridge_body
    
    return ridge_zone


def compute_contact_area(
    ship_vertices: np.ndarray,
    ship_position: Vec2d,
    ship_heading: float,
    ridge_zone: RidgeZone
) -> float:
    """
    Calculate contact area between ship and ridge using polygon intersection.
    
    Args:
        ship_vertices: Ship vertices in local frame (n x 2)
        ship_position: Ship position in world frame
        ship_heading: Ship heading angle (radians)
        ridge_zone: Ridge zone to check contact with
    
    Returns:
        Contact area in m²
    """
    import warnings
    
    # Suppress Shapely warnings for invalid intersections (non-fatal)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*intersection.*')
        
        try:
            from shapely.geometry import Polygon
            from ship_ice_planner.geometry.utils import Rxy
            
            # Validate inputs
            if not hasattr(ridge_zone, 'position') or not hasattr(ridge_zone, 'width') or not hasattr(ridge_zone, 'length'):
                return 0.0
            
            if ridge_zone.width <= 0 or ridge_zone.length <= 0:
                return 0.0
            
            # Check for NaN or invalid values
            if (np.isnan(ship_position.x) or np.isnan(ship_position.y) or
                np.isnan(ridge_zone.position.x) or np.isnan(ridge_zone.position.y) or
                np.isnan(ship_heading) or np.any(np.isnan(ship_vertices))):
                return 0.0
            
            # Transform ship vertices to world frame
            R = Rxy(ship_heading)
            ship_world_vertices = ship_vertices @ R.T + np.array([ship_position.x, ship_position.y])
            
            # Check for invalid vertices
            if np.any(np.isnan(ship_world_vertices)) or np.any(np.isinf(ship_world_vertices)):
                return 0.0
            
            # Create ridge rectangle in world frame
            half_width = ridge_zone.width / 2
            half_length = ridge_zone.length / 2
            ridge_vertices = np.array([
                [ridge_zone.position.x - half_width, ridge_zone.position.y - half_length],
                [ridge_zone.position.x + half_width, ridge_zone.position.y - half_length],
                [ridge_zone.position.x + half_width, ridge_zone.position.y + half_length],
                [ridge_zone.position.x - half_width, ridge_zone.position.y + half_length]
            ])
            
            # Check for invalid ridge vertices
            if np.any(np.isnan(ridge_vertices)) or np.any(np.isinf(ridge_vertices)):
                return 0.0
            
            # Create shapely polygons with validation
            try:
                ship_poly = Polygon(ship_world_vertices)
                if not ship_poly.is_valid or ship_poly.is_empty:
                    ship_poly = ship_poly.buffer(0)
                    if ship_poly.is_empty:
                        return 0.0
            except Exception:
                return 0.0
            
            try:
                ridge_poly = Polygon(ridge_vertices)
                if not ridge_poly.is_valid or ridge_poly.is_empty:
                    ridge_poly = ridge_poly.buffer(0)
                    if ridge_poly.is_empty:
                        return 0.0
            except Exception:
                return 0.0
            
            # Calculate intersection area
            try:
                intersection = ship_poly.intersection(ridge_poly)
                if intersection.is_empty:
                    return 0.0
                
                area = intersection.area
                if np.isnan(area) or np.isinf(area) or area < 0:
                    return 0.0
                
                return area
            except Exception:
                # Intersection failed, use fallback
                pass
        except Exception:
            # Any other error, use fallback
            pass
    
    # Fallback: simple rectangular approximation
    try:
        relative_pos = ship_position - ridge_zone.position
        if (abs(relative_pos.x) < ridge_zone.width / 2 and 
            abs(relative_pos.y) < ridge_zone.length / 2):
            # Approximate contact area as ship width * overlap length
            # This is a rough approximation when shapely fails
            ship_width = np.max(ship_vertices[:, 1]) - np.min(ship_vertices[:, 1])
            ship_length = np.max(ship_vertices[:, 0]) - np.min(ship_vertices[:, 0])
            overlap_length = min(ridge_zone.length, ship_length)
            return ship_width * overlap_length * 0.5  # Conservative estimate
    except Exception:
        pass
    
    return 0.0


def compute_ridge_resistance(
    ship_velocity: Vec2d,
    ship_position: Vec2d,
    ship_vertices: np.ndarray,
    ship_heading: float,
    ridge_zones: List[RidgeZone],
    dt: float,
    current_time: float = None
) -> Tuple[float, float, float]:
    """
    Compute additional resistance from ice ridges using realistic physics model.
    
    New model accounts for:
    - Ridge keel depth (underwater drag)
    - Contact area between ship and ridge
    - Ice compressive strength
    
    Args:
        ship_velocity: Ship velocity vector
        ship_position: Ship position vector
        ship_vertices: Ship vertices in local frame (n x 2)
        ship_heading: Ship heading angle (radians)
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
    
    # Ensure ship_vertices is a numpy array with shape (n, 2)
    if ship_vertices is None:
        return (0.0, 0.0, 0.0)
    
    # Convert to numpy array if needed
    ship_vertices = np.asarray(ship_vertices, dtype=np.float64)
    
    # Ensure 2D shape (n, 2) - vertices as rows
    if ship_vertices.ndim == 1:
        ship_vertices = ship_vertices.reshape(-1, 2)
    elif ship_vertices.ndim == 2:
        if ship_vertices.shape[0] == 2 and ship_vertices.shape[1] > 2:
            # If shape is (2, n), transpose it to (n, 2)
            ship_vertices = ship_vertices.T
        elif ship_vertices.shape[1] != 2 and ship_vertices.shape[0] > 2:
            # Invalid shape, return zero resistance
            return (0.0, 0.0, 0.0)
    
    # Final check: ensure it's a 2D array with shape (n, 2)
    if ship_vertices.ndim != 2 or ship_vertices.shape[1] != 2:
        return (0.0, 0.0, 0.0)
    
    total_resistance = np.array([0.0, 0.0, 0.0])  # Fx, Fy, Mz
    
    for ridge_zone in ridge_zones:
        # Calculate contact area between ship and ridge
        contact_area = compute_contact_area(
            ship_vertices, ship_position, ship_heading, ridge_zone
        )
        
        # Smooth contact area threshold to reduce discrete spikes
        # Use a gradual transition instead of binary on/off
        if contact_area < 0.1:  # No significant contact
            continue
        
        # Smooth contact factor: gradually increase from 0 to 1 as contact area increases
        # This reduces discrete spikes when contact area jumps
        min_contact = 0.1  # m²
        max_contact = 10.0  # m² - full effect at this contact area
        contact_factor = min(1.0, max(0.0, (contact_area - min_contact) / (max_contact - min_contact)))
        
        # Enhanced physically-based resistance model
        # Based on ice mechanics: crushing, bending, and drag components
        
        # 1. ICE CRUSHING RESISTANCE (dominant at low speeds, contact-based)
        # Crushing force: F_crush = σ_crush * A_contact * packing_factor
        # where σ_crush is ice crushing strength, A_contact is contact area
        # NOTE: Use a much smaller fraction of compressive strength (ice doesn't crush uniformly)
        crushing_strength = ICE_COMPRESSIVE_STRENGTH * ridge_zone.packing_density  # Consolidated ice is stronger
        # Scale down significantly: only a small fraction of contact area experiences full crushing
        # Typical ice crushing resistance is ~100-500 kN for large ships, not millions of N
        crushing_area_factor = min(contact_area / 100.0, 1.0)  # Normalize to typical ship area
        crushing_resistance = crushing_strength * contact_area * 0.0001 * crushing_area_factor * contact_factor  # Apply smooth contact factor
        
        # 2. ICE BENDING/BREAKING RESISTANCE (speed-dependent, thickness-dependent)
        # Bending resistance: F_bend = C_bend * thickness^2 * v^α * A_contact
        # where α ≈ 1.5 for ice breaking, thickness is sail height
        # Scale down coefficient significantly
        bending_coefficient = 50.0  # N·s^1.5/m^4 (reduced from 2000)
        velocity_exponent = 1.5  # Non-linear speed dependence (realistic for ice breaking)
        # Scale contact area to prevent huge values
        contact_area_normalized = min(contact_area / 50.0, 2.0)  # Normalize to typical ship area ~50 m²
        bending_resistance = (bending_coefficient * 
                             (ridge_zone.thickness ** 2) * 
                             (ship_speed ** velocity_exponent) * 
                             contact_area_normalized * contact_factor)  # Apply smooth contact factor
        
        # 3. KEEL DRAG RESISTANCE (underwater drag from keel)
        # Drag from keel: F_drag = 0.5 * ρ_water * C_d * A_keel * v²
        # where C_d is drag coefficient, A_keel is keel frontal area
        drag_coefficient = 0.8  # Typical drag coefficient for submerged ice
        # Approximate keel frontal area as keel depth * contact width
        ship_width = np.max(ship_vertices[:, 1]) - np.min(ship_vertices[:, 1])
        # Keel area should be much smaller - only a fraction of keel depth contributes
        keel_frontal_area = ridge_zone.keel_depth * min(ship_width, ridge_zone.width) * 0.1  # Reduced from 0.5
        keel_drag = 0.5 * WATER_DENSITY * drag_coefficient * keel_frontal_area * (ship_speed ** 2) * contact_factor  # Apply smooth contact factor
        
        # 4. ICE SUBMERSION RESISTANCE (buoyancy and ice-water interface)
        # Submersion resistance: F_sub = ρ_water * g * V_displaced * friction_factor
        # Simplified: scales with keel depth and contact area
        submersion_coefficient = 20.0  # N·s²/m⁴ (reduced from 500)
        submersion_resistance = submersion_coefficient * ridge_zone.keel_depth * contact_area_normalized * (ship_speed ** 1.2) * contact_factor  # Apply smooth contact factor
        
        # Combine resistance components
        # At low speeds: crushing dominates
        # At high speeds: bending and drag dominate
        speed_threshold = 1.0  # m/s - transition speed
        if ship_speed < speed_threshold:
            # Low speed: crushing + submersion + reduced bending/drag
            base_resistance = (crushing_resistance * 0.5 +  # Reduced from 0.7
                             bending_resistance * 0.2 * (ship_speed / speed_threshold) ** 0.5 +  # Reduced from 0.3
                             keel_drag * 0.3 +  # Reduced from 0.5
                             submersion_resistance * 0.5)  # Reduced from 0.8
        else:
            # High speed: bending + drag dominate, reduced crushing
            base_resistance = (crushing_resistance * 0.1 +  # Reduced from 0.3
                             bending_resistance * 0.6 +  # Reduced from 1.0
                             keel_drag * 0.5 +  # Reduced from 1.0
                             submersion_resistance * 0.4)  # Reduced from 1.0
        
        # Apply maturity and decay factors (from existing model)
        maturity_factor = ridge_zone.maturity
        # Decay factor: ridges lose effectiveness when inactive
        if current_time is not None:
            if ridge_zone.is_active:
                decay_factor = 1.0
            else:
                inactive_time = current_time - ridge_zone.last_active_time
                decay_factor = max(0.0, 1.0 - inactive_time / RIDGE_DECAY_TIME)
        else:
            # Fallback if time not provided
            decay_factor = 1.0 if ridge_zone.is_active else 0.5
        
        # Packing density factor: consolidated ice is harder to break
        packing_strength_factor = 0.6 + 0.4 * ridge_zone.packing_density  # 0.6 to 1.0 multiplier
        
        # Total resistance magnitude
        resistance_magnitude = (
            base_resistance 
            * maturity_factor 
            * decay_factor
            * packing_strength_factor
        )
        
        # Cap maximum resistance per ridge to prevent ship from getting stuck
        resistance_magnitude = min(resistance_magnitude, MAX_RIDGE_RESISTANCE)
        
        # Resistance opposes motion
        resistance_force = -ship_direction * resistance_magnitude
        total_resistance[0] += resistance_force.x
        total_resistance[1] += resistance_force.y
        
        # Torque: resistance creates moment if contact is off-center
        # Simplified: assume contact center is at ridge position
        contact_offset = ridge_zone.position - ship_position
        moment_arm = contact_offset.length
        if moment_arm > 0.1:  # Only if significant offset
            # Torque = F × r × sin(angle)
            torque = resistance_magnitude * moment_arm * 0.1  # Small torque factor
            total_resistance[2] += torque * np.sign(contact_offset.cross(ship_direction))
    
    # Cap TOTAL resistance from all ridges combined (not just per-ridge)
    # This prevents multiple ridges from adding up to unrealistic values
    total_resistance_magnitude = np.linalg.norm([total_resistance[0], total_resistance[1]])
    if total_resistance_magnitude > MAX_TOTAL_RIDGE_RESISTANCE:
        # Scale down proportionally
        scale_factor = MAX_TOTAL_RIDGE_RESISTANCE / total_resistance_magnitude
        total_resistance[0] *= scale_factor
        total_resistance[1] *= scale_factor
        total_resistance[2] *= scale_factor
    
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

