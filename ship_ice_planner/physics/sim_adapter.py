"""
Compatibility adapter between Chrono physics and sim2d interface.

This module provides ChronoSimAdapter which maintains the same interface
as the existing Pymunk-based simulation in sim2d.py, allowing a smooth
transition to the new physics backend.

The adapter handles:
- Ice field loading from polygon data
- Ship state management
- Control input application
- Observable ice extraction for planners
- Fracture event tracking
"""

from typing import Optional, Union
import numpy as np
from shapely.geometry import Polygon

from .ice_material import IceMaterialParams
from .chrono_backend import ChronoIceSimulator, FloeType
from .bonded_assembly import BondedAssembly


class ChronoSimAdapter:
    """
    Adapter providing sim2d-compatible interface to ChronoIceSimulator.
    
    This adapter allows the new PyChrono backend to be used as a drop-in
    replacement for the Pymunk physics in sim2d.py.
    
    Attributes:
        simulator: Underlying ChronoIceSimulator
        config: Configuration dictionary
        dt: Time step for simulation
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Chrono simulation adapter.
        
        Args:
            config: Configuration dictionary with physics, ice_material,
                   bonded_dem, and ship parameters
        """
        self.config = config
        
        # Extract physics config
        physics_cfg = config.get('physics', {})
        self.dt = physics_cfg.get('dt', 0.02)
        use_gpu = physics_cfg.get('use_gpu', False)
        
        # Create material from config
        self.material = IceMaterialParams.from_config(config)
        
        # Create simulator
        self.simulator = ChronoIceSimulator(
            material=self.material,
            dt=self.dt,
            use_gpu=use_gpu,
            config=config,
        )
        
        # Bonded DEM settings
        bonded_cfg = config.get('bonded_dem', {})
        self.min_area_for_bonding = bonded_cfg.get('min_floe_area_for_bonding', 500.0)
        self.particle_size_ratio = bonded_cfg.get('particle_size_ratio', 0.08)
        
        # Ship config
        ship_cfg = config.get('ship', {})
        self.ship_mass = ship_cfg.get('mass', 10000.0)
        self.ship_length = ship_cfg.get('length', 50.0)
        self.ship_beam = ship_cfg.get('beam', 10.0)
        
        # Track floe ID to polygon mapping for planners
        self._floe_polygons: dict[int, np.ndarray] = {}
        self._floe_thicknesses: dict[int, float] = {}
        
        # Current control inputs
        self._thrust = 0.0
        self._rudder = 0.0
        
        # Collision tracking for analysis
        self._collision_impulses: list[dict] = []
        
    def load_ice_field(self, 
                       polygons: list[np.ndarray], 
                       thicknesses: Optional[list[float]] = None,
                       use_bonded: Optional[list[bool]] = None):
        """
        Load ice floes into the simulation.
        
        Args:
            polygons: List of Nx2 vertex arrays for each floe
            thicknesses: Optional thickness for each floe (default: 1.0m)
            use_bonded: Optional flag for each floe to use bonded DEM
                       (if None, determined by area threshold)
        """
        n_floes = len(polygons)
        
        # Default thicknesses
        if thicknesses is None:
            thicknesses = [self.material.thickness] * n_floes
            
        # Determine which floes should be bonded
        if use_bonded is None:
            use_bonded = []
            for poly_verts in polygons:
                poly = Polygon(poly_verts)
                area = poly.area
                # Large floes get bonded DEM, small ones stay rigid
                use_bonded.append(area >= self.min_area_for_bonding)
                
        # Create floes
        for i, (vertices, thickness, bonded) in enumerate(zip(polygons, thicknesses, use_bonded)):
            vertices = np.asarray(vertices, dtype=float)
            
            if bonded:
                floe_id = self.simulator.create_ice_floe_bonded(
                    vertices=vertices,
                    thickness=thickness,
                )
            else:
                floe_id = self.simulator.create_ice_floe_rigid(
                    vertices=vertices,
                    thickness=thickness,
                )
                
            # Store polygon for later retrieval
            self._floe_polygons[floe_id] = vertices
            self._floe_thicknesses[floe_id] = thickness
            
    def set_ship_pose(self, x: float, y: float, heading: float):
        """
        Set ship initial position and heading.
        
        Args:
            x: X position in meters
            y: Y position in meters  
            heading: Heading angle in radians (0 = +x direction)
        """
        self.simulator.create_ship(
            position=(x, y),
            heading=heading,
            mass=self.ship_mass,
            length=self.ship_length,
            beam=self.ship_beam,
        )
        
    def apply_control(self, thrust: float, rudder: float):
        """
        Set control inputs for next simulation step.
        
        Args:
            thrust: Forward thrust force in N
            rudder: Rudder torque in N·m
        """
        self._thrust = thrust
        self._rudder = rudder
        
    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step (uses default if None)
        """
        # Apply current control
        self.simulator.apply_ship_thrust(self._thrust, self._rudder)
        
        # Step physics
        self.simulator.step(dt)
        
    def get_ship_state(self) -> np.ndarray:
        """
        Get current ship state as numpy array.
        
        Returns:
            Array [x, y, psi, u, v, r] where:
            - x, y: Position in meters
            - psi: Heading in radians
            - u: Surge velocity (forward) in m/s
            - v: Sway velocity (lateral) in m/s
            - r: Yaw rate in rad/s
        """
        state_dict = self.simulator.get_ship_state()
        
        # Convert world-frame velocities to body-frame
        vx = state_dict['vx']
        vy = state_dict['vy']
        psi = state_dict['heading']
        
        # Rotation matrix from world to body frame
        c, s = np.cos(psi), np.sin(psi)
        u = c * vx + s * vy    # Surge (forward)
        v = -s * vx + c * vy   # Sway (lateral)
        
        return np.array([
            state_dict['x'],
            state_dict['y'],
            psi,
            u,
            v,
            state_dict['omega']
        ])
    
    def get_ship_position(self) -> tuple[float, float]:
        """Get ship (x, y) position."""
        state = self.simulator.get_ship_state()
        return (state['x'], state['y'])
    
    def get_observable_ice(self, 
                           ship_pos: Optional[tuple[float, float]] = None,
                           range_limit: float = float('inf')) -> list[dict]:
        """
        Get ice floes observable from ship position.
        
        Args:
            ship_pos: Ship position (uses current if None)
            range_limit: Maximum range to observe floes
            
        Returns:
            List of floe dictionaries with vertices, centroid, etc.
        """
        if ship_pos is None:
            ship_pos = self.get_ship_position()
            
        ship_pos = np.array(ship_pos)
        observable = []
        
        for floe_id, floe in self.simulator.ice_floes.items():
            # Get floe centroid
            if floe.is_bonded and floe.assembly:
                centroid = floe.assembly.centroid
            else:
                poly = Polygon(floe.vertices)
                centroid = np.array(poly.centroid.coords[0])
                
            # Check range
            dist = np.linalg.norm(centroid - ship_pos)
            if dist > range_limit:
                continue
                
            # Get current vertices (may have moved)
            vertices = self._get_floe_vertices(floe_id)
            
            observable.append({
                'id': floe_id,
                'vertices': vertices,
                'centroid': centroid,
                'thickness': self._floe_thicknesses.get(floe_id, 1.0),
                'is_bonded': floe.is_bonded,
                'n_fragments': floe.n_fragments,
                'distance': dist,
            })
            
        # Sort by distance
        observable.sort(key=lambda x: x['distance'])
        
        return observable
    
    def _get_floe_vertices(self, floe_id: int) -> np.ndarray:
        """
        Get current vertices for a floe (accounting for movement).
        
        For bonded floes after fracture, returns convex hull of fragments.
        """
        floe = self.simulator.ice_floes.get(floe_id)
        if floe is None:
            return self._floe_polygons.get(floe_id, np.array([]))
            
        if floe.is_bonded and floe.assembly:
            # For bonded floes, get convex hull of all particles
            if self.simulator._mock_mode:
                # In mock mode, use original vertices
                return floe.vertices
            else:
                # Get particle positions from Chrono bodies
                positions = []
                for body in floe.particles:
                    pos = body.GetPos()
                    positions.append([pos.x, pos.y])
                    
                if len(positions) >= 3:
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(positions)
                        return np.array(positions)[hull.vertices]
                    except Exception:
                        pass
                        
        # Return original or rigid body vertices
        return floe.vertices
    
    def get_fracture_events(self) -> list[dict]:
        """
        Get fracture events since last call.
        
        Returns:
            List of fracture event dictionaries with time, location,
            failure mode, etc.
        """
        return self.simulator.get_fracture_events()
    
    def get_collision_impulses(self) -> list[dict]:
        """Get collision impulses since last call."""
        impulses = self._collision_impulses.copy()
        self._collision_impulses.clear()
        return impulses
    
    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.simulator.time
    
    def get_total_floe_count(self) -> int:
        """Get total number of ice floes."""
        return len(self.simulator.ice_floes)
    
    def get_fragment_count(self) -> int:
        """Get total number of fragments across all bonded floes."""
        count = 0
        for floe in self.simulator.ice_floes.values():
            if floe.is_bonded:
                count += floe.n_fragments
            else:
                count += 1
        return count
    
    def get_energy_state(self) -> dict:
        """
        Get energy metrics for the simulation.
        
        Returns:
            Dictionary with kinetic_energy, etc.
        """
        return {
            'kinetic_energy': self.simulator.get_total_kinetic_energy(),
            'time': self.simulator.time,
        }
    
    def cleanup(self):
        """Clean up simulation resources."""
        self.simulator.cleanup()
        self._floe_polygons.clear()
        self._floe_thicknesses.clear()
        
    @classmethod
    def from_config_file(cls, config_path: str) -> 'ChronoSimAdapter':
        """
        Create adapter from YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Configured ChronoSimAdapter instance
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)


def run_comparison_test(pymunk_adapter, chrono_adapter, 
                        duration: float = 30.0,
                        dt: float = 0.02) -> dict:
    """
    Run comparison test between Pymunk and Chrono backends.
    
    Args:
        pymunk_adapter: Pymunk-based simulation adapter
        chrono_adapter: ChronoSimAdapter instance
        duration: Test duration in seconds
        dt: Time step
        
    Returns:
        Dictionary with comparison metrics
    """
    pymunk_trajectory = []
    chrono_trajectory = []
    
    steps = int(duration / dt)
    
    for _ in range(steps):
        # Step both
        pymunk_adapter.step(dt)
        chrono_adapter.step(dt)
        
        # Record positions
        pymunk_pos = pymunk_adapter.get_ship_position()
        chrono_pos = chrono_adapter.get_ship_position()
        
        pymunk_trajectory.append(pymunk_pos)
        chrono_trajectory.append(chrono_pos)
        
    pymunk_traj = np.array(pymunk_trajectory)
    chrono_traj = np.array(chrono_trajectory)
    
    # Compute metrics
    position_diff = np.linalg.norm(pymunk_traj - chrono_traj, axis=1)
    
    return {
        'pymunk_trajectory': pymunk_traj,
        'chrono_trajectory': chrono_traj,
        'position_diff': position_diff,
        'mean_diff': np.mean(position_diff),
        'max_diff': np.max(position_diff),
        'final_diff': position_diff[-1],
    }
