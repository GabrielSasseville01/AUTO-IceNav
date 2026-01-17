"""
Pymunk physics backend wrapper.

Wraps the existing Pymunk simulation code to provide a unified interface
compatible with the ChronoSimAdapter, enabling backend switching in sim2d.py.
"""

import numpy as np
import pymunk
from pymunk import Vec2d
from typing import List, Dict, Optional, Tuple, Any

from ship_ice_planner.utils.sim_utils import (
    init_pymunk_space,
    create_sim_ship,
    create_polygon,
    generate_sim_obs,
    ICE_DENSITY,
    ICE_THICKNESS,
    SHIP_MASS,
)
from ship_ice_planner.geometry.polygon import poly_area


class PymunkBackend:
    """
    Pymunk physics backend providing unified interface for sim2d.py.
    
    This wrapper exposes the same interface as ChronoSimAdapter,
    allowing seamless backend switching based on configuration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Pymunk backend.
        
        Args:
            config: Configuration dictionary (same format as sim2d config)
        """
        self.config = config
        self.space = init_pymunk_space()
        
        # Ship state
        self.ship_shape: Optional[pymunk.Poly] = None
        self.ship_body: Optional[pymunk.Body] = None
        
        # Ice floe storage
        self.ice_shapes: List[pymunk.Poly] = []
        self._floe_polygons: Dict[int, np.ndarray] = {}
        
        # Collision tracking
        self._collision_events: List[Dict] = []
        self._setup_collision_handler()
        
        # Time tracking
        self.time = 0.0
        self.dt = config.get('sim_dynamics', {}).get('dt', 0.02)
        
    def _setup_collision_handler(self):
        """Setup collision handlers for ship-ice collisions."""
        def pre_solve_handler(arbiter, space, data):
            # Store pre-collision KE for analysis
            ice_shape = arbiter.shapes[1]  # Assuming ship=1, ice=2
            ice_shape.body.pre_collision_KE = ice_shape.body.kinetic_energy
            return True
            
        def post_solve_handler(arbiter, space, data):
            ship_shape, ice_shape = arbiter.shapes
            
            # Record collision event
            self._collision_events.append({
                'time': self.time,
                'impulse': arbiter.total_impulse,
                'total_ke': arbiter.total_ke,
                'ice_idx': getattr(ice_shape, 'idx', -1),
                'contact_points': [
                    list(ship_shape.body.world_to_local(cp.point_b))
                    for cp in arbiter.contact_point_set.points
                ],
            })
        
        # Register handlers for ship (type 1) vs ice (type 2)
        self.space.on_collision(
            collision_type_a=1,
            collision_type_b=2,
            pre_solve=pre_solve_handler,
            post_solve=post_solve_handler
        )
        
    def load_ice_field(self, 
                       polygons: List[np.ndarray],
                       thicknesses: Optional[List[float]] = None,
                       obs_dicts: Optional[List[Dict]] = None):
        """
        Load ice floes into the simulation.
        
        Args:
            polygons: List of Nx2 vertex arrays
            thicknesses: Optional thickness for each floe (ignored in 2D Pymunk)
            obs_dicts: Optional obstacle dictionaries with centre info
        """
        self.ice_shapes.clear()
        self._floe_polygons.clear()
        
        if obs_dicts is not None:
            # Use full obstacle dicts if provided
            self.ice_shapes = generate_sim_obs(self.space, obs_dicts)
        else:
            # Create floes from raw polygons
            for i, verts in enumerate(polygons):
                verts = np.asarray(verts, dtype=float)
                centroid = np.mean(verts, axis=0)
                local_verts = verts - centroid
                
                shape = create_polygon(
                    self.space,
                    local_verts.tolist(),
                    centroid[0], centroid[1]
                )
                self.ice_shapes.append(shape)
        
        # Set collision types and indices
        for idx, shape in enumerate(self.ice_shapes):
            shape.collision_type = 2  # Ice collision type
            shape.idx = idx
            self._floe_polygons[idx] = polygons[idx] if idx < len(polygons) else None
            
    def create_ship(self, 
                    vertices: np.ndarray,
                    position: Tuple[float, float, float],
                    mass: float = SHIP_MASS) -> pymunk.Poly:
        """
        Create the ship body.
        
        Args:
            vertices: Ship hull vertices
            position: (x, y, heading) initial pose
            mass: Ship mass in kg
            
        Returns:
            Pymunk shape for the ship
        """
        self.ship_shape = create_sim_ship(
            self.space,
            vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
            position,
            body_type=pymunk.Body.KINEMATIC
        )
        self.ship_shape.collision_type = 1  # Ship collision type
        self.ship_body = self.ship_shape.body
        
        return self.ship_shape
        
    def set_ship_pose(self, x: float, y: float, heading: float):
        """Set ship position and heading."""
        if self.ship_body:
            self.ship_body.position = Vec2d(x, y)
            self.ship_body.angle = heading
            
    def set_ship_velocity(self, vx: float, vy: float, omega: float):
        """Set ship velocity (for kinematic control)."""
        if self.ship_body:
            self.ship_body.velocity = Vec2d(vx, vy)
            self.ship_body.angular_velocity = omega
            
    def apply_control(self, thrust: float, rudder: float):
        """
        Apply control forces to ship.
        
        Note: For kinematic ships, this is ignored. Use set_ship_velocity instead.
        For dynamic ships, this would apply forces.
        """
        # Kinematic ships don't respond to forces
        # This is here for interface compatibility with Chrono
        pass
        
    def step(self, dt: Optional[float] = None):
        """Advance simulation by one time step."""
        dt = dt or self.dt
        self.space.step(dt)
        self.time += dt
        
    def get_ship_state(self) -> np.ndarray:
        """
        Get ship state vector.
        
        Returns:
            Array [x, y, psi, vx, vy, omega]
        """
        if self.ship_body is None:
            return np.zeros(6)
            
        pos = self.ship_body.position
        vel = self.ship_body.velocity
        
        return np.array([
            pos.x, pos.y, self.ship_body.angle,
            vel.x, vel.y, self.ship_body.angular_velocity
        ])
        
    def get_ice_states(self) -> List[Dict]:
        """
        Get states of all ice floes.
        
        Returns:
            List of dicts with position, velocity, vertices for each floe
        """
        states = []
        for idx, shape in enumerate(self.ice_shapes):
            body = shape.body
            pos = body.position
            vel = body.velocity
            
            # Get transformed vertices
            local_verts = np.array([[v.x, v.y] for v in shape.get_vertices()])
            
            states.append({
                'id': idx,
                'x': pos.x,
                'y': pos.y,
                'angle': body.angle,
                'vx': vel.x,
                'vy': vel.y,
                'omega': body.angular_velocity,
                'vertices': local_verts,
                'mass': body.mass,
            })
            
        return states
        
    def get_collision_events(self) -> List[Dict]:
        """Get collision events since last call."""
        events = self._collision_events.copy()
        self._collision_events.clear()
        return events
        
    def get_fracture_events(self) -> List[Dict]:
        """
        Get fracture events.
        
        Note: Pymunk doesn't support fracturing. Returns empty list.
        For fracturing, use the Chrono backend.
        """
        return []
        
    def get_energy_state(self) -> Dict[str, float]:
        """Get energy metrics."""
        ke = 0.0
        
        # Ship KE
        if self.ship_body:
            ke += self.ship_body.kinetic_energy
            
        # Ice KE
        for shape in self.ice_shapes:
            ke += shape.body.kinetic_energy
            
        return {
            'kinetic_energy': ke,
            'time': self.time,
        }
        
    def get_total_floe_count(self) -> int:
        """Get number of ice floes."""
        return len(self.ice_shapes)
        
    def cleanup(self):
        """Clean up Pymunk resources."""
        # Remove all bodies and shapes from space
        for shape in self.ice_shapes:
            if shape.body in self.space.bodies:
                self.space.remove(shape.body, shape)
                
        if self.ship_shape and self.ship_body:
            if self.ship_body in self.space.bodies:
                self.space.remove(self.ship_body, self.ship_shape)
                
        self.ice_shapes.clear()
        self._floe_polygons.clear()
        self.ship_shape = None
        self.ship_body = None
        self._collision_events.clear()
        
    @property
    def current_time(self) -> float:
        """Current simulation time."""
        return self.time
