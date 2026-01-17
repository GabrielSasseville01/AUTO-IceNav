"""
PyChrono-based physics simulator for ice floe simulation.

This module provides the core physics simulation using Project Chrono,
supporting both rigid ice floes and bonded DEM assemblies that can fracture.

Key features:
- Rigid body dynamics for ship and ice floes
- Bonded particle assemblies with breakable bonds
- Contact/collision detection and response
- Buoyancy and hydrodynamic damping
- Bond failure monitoring and fragment tracking
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from enum import Enum
import warnings

try:
    import pychrono as chrono
    import pychrono.core as chrono_core
    CHRONO_AVAILABLE = True
except ImportError:
    CHRONO_AVAILABLE = False
    warnings.warn(
        "PyChrono not installed. Install via: conda install -c conda-forge pychrono\n"
        "The ChronoIceSimulator will run in mock mode for testing."
    )

from .ice_material import IceMaterialParams
from .bonded_assembly import BondedAssembly, Bond, Particle


class FloeType(Enum):
    """Type of ice floe representation."""
    RIGID = "rigid"
    BONDED = "bonded"


@dataclass
class IceFloe:
    """
    Container for ice floe data in the simulation.
    
    Can represent either a simple rigid body or a bonded particle assembly.
    
    Attributes:
        id: Unique floe identifier
        floe_type: RIGID or BONDED
        vertices: Original polygon vertices
        thickness: Ice thickness in meters
        body: Chrono rigid body (for RIGID type)
        assembly: BondedAssembly (for BONDED type)
        particles: List of Chrono bodies (for BONDED type)
        bonds: List of bond objects (for BONDED type)
    """
    id: int
    floe_type: FloeType
    vertices: np.ndarray
    thickness: float = 1.0
    
    # For RIGID floes
    body: Optional[object] = None  # chrono.ChBody
    
    # For BONDED floes
    assembly: Optional[BondedAssembly] = None
    particles: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    
    # Derived properties
    particle_radius: float = 1.0
    
    @property
    def is_bonded(self) -> bool:
        return self.floe_type == FloeType.BONDED
    
    @property
    def n_particles(self) -> int:
        if self.assembly:
            return self.assembly.n_particles
        return 0
    
    @property
    def n_broken_bonds(self) -> int:
        if self.assembly:
            return self.assembly.n_broken_bonds
        return 0
    
    @property
    def n_fragments(self) -> int:
        if self.assembly:
            return len(self.assembly.get_fragments())
        return 1


class ChronoIceSimulator:
    """
    PyChrono-based simulator for ship-ice interaction with fracture.
    
    Manages a Chrono system with:
    - Ship rigid body with applied forces
    - Ice floes (rigid or bonded DEM)
    - Contact detection and response
    - Bond failure monitoring
    - Buoyancy forces
    
    Attributes:
        system: Chrono system object
        material: Ice material parameters
        ship_body: Ship rigid body
        ice_floes: Dictionary of IceFloe objects
        dt: Time step size
        time: Current simulation time
    """
    
    # Water properties
    WATER_DENSITY = 1025.0  # kg/m³
    WATER_LEVEL = 0.0       # z-coordinate of water surface
    GRAVITY = 9.81          # m/s²
    
    def __init__(self, 
                 material: Optional[IceMaterialParams] = None,
                 dt: float = 0.01,
                 use_gpu: bool = False,
                 config: Optional[dict] = None):
        """
        Initialize the Chrono ice simulator.
        
        Args:
            material: Ice material parameters
            dt: Time step for simulation
            use_gpu: Whether to use GPU acceleration (requires Chrono::GPU)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        # Create material from config if not provided directly
        if material is not None:
            self.material = material
        else:
            self.material = IceMaterialParams.from_config(self.config)
        
        # Use dt from config if provided, otherwise use parameter
        physics_cfg = self.config.get('physics', {})
        self.dt = physics_cfg.get('dt', dt)
        self.use_gpu = physics_cfg.get('use_gpu', use_gpu)
        self.time = 0.0
        
        # Floe storage
        self.ice_floes: dict[int, IceFloe] = {}
        self._floe_counter = 0
        
        # Ship reference
        self.ship_body: Optional[object] = None
        self._ship_state = np.zeros(6)  # [x, y, psi, u, v, r]
        
        # Fracture event tracking
        self.fracture_events: list[dict] = []
        self._collision_impulses: list[dict] = []
        
        # Initialize Chrono system
        self._init_chrono_system()
        
    def _init_chrono_system(self):
        """Initialize the Chrono physics system."""
        if not CHRONO_AVAILABLE:
            # Mock mode for testing without Chrono
            self.system = None
            self._mock_mode = True
            return
            
        self._mock_mode = False
        
        # Create Chrono system
        # Use SMC (smooth contact) for better stability with bonded particles
        self.system = chrono.ChSystemSMC()
        
        # DISABLE gravity for 2D XY simulation
        # Bodies are constrained to Z=0 plane; gravity/buoyancy handled separately
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))
        
        # Configure solver
        self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.system.GetSolver().AsIterative().SetMaxIterations(100)
        
        # Configure collision detection
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        
        # Create contact material with proper stiffness for stability
        # NOTE: Using CONTACT stiffness (1e7 Pa), NOT bulk Young's modulus (5e9 Pa)
        # High bulk modulus causes numerical instability with typical time steps
        self._contact_material = chrono.ChContactMaterialSMC()
        self._contact_material.SetFriction(self.material.friction)
        self._contact_material.SetRestitution(0.1)  # Low for energy dissipation
        
        # Contact stiffness - much lower than bulk modulus for stability
        self._contact_material.SetYoungModulus(1e7)  # 10 MPa contact stiffness
        self._contact_material.SetPoissonRatio(0.3)
        
        # SMC-specific damping coefficients for numerical stability
        self._contact_material.SetKn(1e6)   # Normal contact stiffness
        self._contact_material.SetKt(1e5)   # Tangential contact stiffness  
        self._contact_material.SetGn(5e3)   # Normal damping coefficient
        self._contact_material.SetGt(5e3)   # Tangential damping coefficient
        
    def create_ship(self, 
                    position: tuple[float, float],
                    heading: float,
                    mass: float = 10000.0,
                    length: float = 50.0,
                    beam: float = 10.0,
                    initial_velocity: tuple[float, float] = (0.0, 0.0)) -> object:
        """
        Create the ship rigid body.
        
        Args:
            position: (x, y) position in meters
            heading: Heading angle in radians (0 = +x direction)
            mass: Ship mass in kg
            length: Ship length in meters
            beam: Ship beam (width) in meters
            initial_velocity: (vx, vy) initial velocity in m/s
            
        Returns:
            Chrono body object (or mock object if Chrono unavailable)
        """
        if self._mock_mode:
            self._ship_state = np.array([
                position[0], position[1], heading,
                initial_velocity[0], initial_velocity[1], 0.0
            ])
            self.ship_body = {'mass': mass, 'length': length, 'beam': beam}
            return self.ship_body
            
        # Create ship body as a box (better for 2D XY collision)
        # Using ChBody instead of ChBodyEasyCylinder for more control
        ship = chrono.ChBody()
        ship.SetMass(mass)
        
        # Inertia for box
        draft = beam * 0.5  # Ship draft
        Ixx = mass * (beam**2 + draft**2) / 12
        Iyy = mass * (length**2 + draft**2) / 12
        Izz = mass * (length**2 + beam**2) / 12
        ship.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
        
        # Set position at Z=0 (same plane as ice for collision)
        ship.SetPos(chrono.ChVector3d(position[0], position[1], 0))
        
        # Set rotation (heading around Z axis)
        q = chrono.QuatFromAngleZ(heading)
        ship.SetRot(q)
        
        # Set initial velocity
        ship.SetPosDt(chrono.ChVector3d(initial_velocity[0], initial_velocity[1], 0))
        
        # Enable collision FIRST
        ship.EnableCollision(True)
        
        # Add collision shape (box representing ship hull)
        box_shape = chrono.ChCollisionShapeBox(
            self._contact_material,
            length, beam, draft  # Full dimensions
        )
        ship.AddCollisionShape(box_shape, chrono.ChFramed())
        
        # Set collision family (PyChrono 9.x - no Build() needed, it's automatic)
        ship.GetCollisionModel().SetFamily(1)  # Ship = family 1
        ship.GetCollisionModel().DisallowCollisionsWith(1)  # No self-collision
        
        # Add to system
        self.system.Add(ship)
        
        self.ship_body = ship
        
        # Update state
        self._update_ship_state()
        
        return ship
    
    def create_ice_floe_rigid(self,
                              vertices: np.ndarray,
                              thickness: float = 1.0,
                              position_offset: tuple[float, float] = (0, 0),
                              initial_velocity: tuple[float, float] = (0, 0)) -> int:
        """
        Create a rigid (non-fracturing) ice floe.
        
        Args:
            vertices: Nx2 array of polygon vertices
            thickness: Ice thickness in meters
            position_offset: (dx, dy) offset from vertex coordinates
            initial_velocity: (vx, vy) initial velocity
            
        Returns:
            Floe ID
        """
        vertices = np.asarray(vertices, dtype=float)
        floe_id = self._floe_counter
        self._floe_counter += 1
        
        if self._mock_mode:
            # Mock floe for testing
            floe = IceFloe(
                id=floe_id,
                floe_type=FloeType.RIGID,
                vertices=vertices,
                thickness=thickness,
            )
            self.ice_floes[floe_id] = floe
            return floe_id
            
        # Compute centroid
        from shapely.geometry import Polygon
        poly = Polygon(vertices)
        centroid = np.array(poly.centroid.coords[0])
        
        # Compute mass
        area = poly.area
        volume = area * thickness
        mass = self.material.density * volume
        
        # Create convex hull mesh for collision
        # Chrono expects vertices relative to body center
        rel_vertices = vertices - centroid
        
        # Create body
        body = chrono.ChBody()
        body.SetMass(mass)
        
        # Approximate inertia as rectangle
        Ixx = mass * (thickness**2 + (np.ptp(vertices[:, 1]))**2) / 12
        Iyy = mass * (thickness**2 + (np.ptp(vertices[:, 0]))**2) / 12
        Izz = mass * (np.ptp(vertices[:, 0])**2 + np.ptp(vertices[:, 1])**2) / 12
        body.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
        
        # Set position (centroid + offset, at Z=0 for 2D XY collision)
        pos = centroid + np.array(position_offset)
        body.SetPos(chrono.ChVector3d(pos[0], pos[1], 0))  # Z=0 same as ship
        
        # Set initial velocity
        body.SetPosDt(chrono.ChVector3d(initial_velocity[0], initial_velocity[1], 0))
        
        # Enable collision FIRST
        body.EnableCollision(True)
        
        # Add collision shape (extruded polygon approximated as box)
        # Use box approximation for collision (PyChrono 9.x API)
        length_x = np.ptp(vertices[:, 0])
        length_y = np.ptp(vertices[:, 1])
        length_z = thickness
        
        box_shape = chrono.ChCollisionShapeBox(
            self._contact_material,
            length_x, length_y, length_z
        )
        body.AddCollisionShape(box_shape, chrono.ChFramed())
        
        # Set collision family (PyChrono 9.x - no Build() needed)
        body.GetCollisionModel().SetFamily(2)  # Ice = family 2
        
        # Add to system
        self.system.Add(body)
        
        # Store floe
        floe = IceFloe(
            id=floe_id,
            floe_type=FloeType.RIGID,
            vertices=vertices,
            thickness=thickness,
            body=body,
        )
        self.ice_floes[floe_id] = floe
        
        return floe_id
    
    def create_ice_floe_bonded(self,
                               vertices: np.ndarray,
                               thickness: float = 1.0,
                               particle_radius: Optional[float] = None,
                               position_offset: tuple[float, float] = (0, 0)) -> int:
        """
        Create a bonded DEM ice floe that can fracture.
        
        Args:
            vertices: Nx2 array of polygon vertices
            thickness: Ice thickness in meters
            particle_radius: Radius of DEM particles (computed from size_ratio if None)
            position_offset: (dx, dy) offset from vertex coordinates
            
        Returns:
            Floe ID
        """
        vertices = np.asarray(vertices, dtype=float)
        floe_id = self._floe_counter
        self._floe_counter += 1
        
        # Get bonded DEM config
        bonded_cfg = self.config.get('bonded_dem', {})
        size_ratio = bonded_cfg.get('particle_size_ratio', 0.08)
        
        # Create bonded assembly
        assembly = BondedAssembly(
            vertices=vertices,
            particle_radius=particle_radius,
            material=self.material,
            size_ratio=size_ratio,
        )
        
        if self._mock_mode:
            # Mock floe for testing
            floe = IceFloe(
                id=floe_id,
                floe_type=FloeType.BONDED,
                vertices=vertices,
                thickness=thickness,
                assembly=assembly,
                particle_radius=assembly.particle_radius,
            )
            self.ice_floes[floe_id] = floe
            return floe_id
            
        # Create Chrono bodies for each particle
        chrono_particles = []
        r = assembly.particle_radius
        
        for particle in assembly.particles:
            # Compute particle mass (already set in assembly)
            pos = particle.pos + np.array(position_offset)
            
            # Create sphere body
            body = chrono.ChBodyEasySphere(
                r,
                self.material.density,
                True,   # visualization
                True,   # collision
                self._contact_material
            )
            
            # Set position at Z=0 for 2D XY collision with ship
            body.SetPos(chrono.ChVector3d(pos[0], pos[1], 0))
            
            # Set collision family (PyChrono 9.x - no Build() needed)
            if body.GetCollisionModel():
                body.GetCollisionModel().SetFamily(2)  # Ice = family 2
            
            # Add to system
            self.system.Add(body)
            chrono_particles.append(body)
            
            # Store reference in particle
            particle.chrono_body = body
            
        # Create bonds between particles using rigid distance constraints
        # These are truly rigid (no spring bounce) and break cleanly when force exceeded
        chrono_bonds = []
        
        # Get bond type from config - 'rigid' for clean fracture, 'spring' for elastic
        bond_type = self.config.get('bonded_dem', {}).get('bond_type', 'rigid')
        
        for bond in assembly.bonds:
            body_a = chrono_particles[bond.i]
            body_b = chrono_particles[bond.j]
            
            pos_a = body_a.GetPos()
            pos_b = body_b.GetPos()
            
            if bond_type == 'spring':
                # Spring-damper for elastic behavior
                link = chrono.ChLinkTSDA()
                link.Initialize(body_a, body_b, False, pos_a, pos_b)
                link.SetSpringCoefficient(bond.k_n)
                link.SetDampingCoefficient(bond.k_n * 0.1)
                link.SetRestLength(bond.rest_length)
            else:
                # Rigid distance constraint for brittle fracture
                link = chrono.ChLinkDistance()
                link.Initialize(body_a, body_b, False, pos_a, pos_b)
                # Set imposed distance (will maintain this rigidly)
                link.SetImposedDistance(bond.rest_length)
            
            # Add to system
            self.system.Add(link)
            chrono_bonds.append(link)
            
            # Store reference and type
            bond.chrono_link = link
            bond.link_type = bond_type
            
        # Store floe
        floe = IceFloe(
            id=floe_id,
            floe_type=FloeType.BONDED,
            vertices=vertices,
            thickness=thickness,
            assembly=assembly,
            particles=chrono_particles,
            bonds=chrono_bonds,
            particle_radius=assembly.particle_radius,
        )
        self.ice_floes[floe_id] = floe
        
        return floe_id
    
    def apply_ship_thrust(self, thrust: float, rudder: float):
        """
        Apply thrust and rudder forces to ship.
        
        Args:
            thrust: Forward thrust force in N
            rudder: Rudder torque in N·m
        """
        if self._mock_mode:
            # Simple mock dynamics
            psi = self._ship_state[2]
            # Thrust in ship frame, transform to world
            fx = thrust * np.cos(psi)
            fy = thrust * np.sin(psi)
            
            # Simple acceleration (F = ma)
            mass = self.ship_body.get('mass', 10000.0)
            ax = fx / mass
            ay = fy / mass
            alpha = rudder / (mass * 10)  # Simplified rotational inertia
            
            # Update velocities
            self._ship_state[3] += ax * self.dt
            self._ship_state[4] += ay * self.dt
            self._ship_state[5] += alpha * self.dt
            return
            
        if self.ship_body is None:
            return
            
        # Get ship heading
        rot = self.ship_body.GetRot()
        heading = rot.GetCardanAnglesZYX().z
        
        # Thrust in world frame
        force = chrono.ChVector3d(
            thrust * np.cos(heading),
            thrust * np.sin(heading),
            0
        )
        self.ship_body.AccumulateForce(force, self.ship_body.GetPos(), False)
        
        # Rudder torque around Z axis
        torque = chrono.ChVector3d(0, 0, rudder)
        self.ship_body.AccumulateTorque(torque, False)
        
    # Maximum stable time step for SMC contacts (2ms)
    MAX_STABLE_DT = 0.002
    
    def step(self, dt: Optional[float] = None, time: Optional[float] = None):
        """
        Advance simulation by one time step with automatic sub-stepping.
        
        Args:
            dt: Time step (uses default if None)
            time: Current simulation time (auto-incremented if None)
        """
        dt = dt or self.dt
        
        if self._mock_mode:
            # Simple mock physics
            self._mock_step(dt)
            self.time += dt
            return
        
        # Sub-stepping for numerical stability
        # Large dt values cause force explosions with stiff contacts
        if dt > self.MAX_STABLE_DT:
            n_substeps = int(np.ceil(dt / self.MAX_STABLE_DT))
            sub_dt = dt / n_substeps
            for _ in range(n_substeps):
                self._do_substep(sub_dt)
        else:
            self._do_substep(dt)
        
        self.time += dt
    
    def _do_substep(self, dt: float):
        """Execute a single physics sub-step."""
        # Apply hydrodynamic damping (XY only)
        self._apply_damping()
        
        # Step Chrono system
        self.system.DoStepDynamics(dt)
        
        # Enforce 2D XY planar constraint (zero out Z motion)
        self._enforce_planar_constraint()
        
        # Check for bond failures
        self._check_bond_failures()
        
        # Update internal state
        self._update_ship_state()
        
    def _enforce_planar_constraint(self):
        """Constrain all bodies to Z=0 plane for 2D simulation."""
        if self._mock_mode:
            return
        
        # Enforce on ship
        if self.ship_body:
            pos = self.ship_body.GetPos()
            self.ship_body.SetPos(chrono.ChVector3d(pos.x, pos.y, 0))
            
            vel = self.ship_body.GetPosDt()
            self.ship_body.SetPosDt(chrono.ChVector3d(vel.x, vel.y, 0))
            
            # Keep rotation only around Z axis
            rot = self.ship_body.GetRot()
            heading = rot.GetCardanAnglesZYX().z
            self.ship_body.SetRot(chrono.QuatFromAngleZ(heading))
            
            omega = self.ship_body.GetAngVelParent()
            self.ship_body.SetAngVelParent(chrono.ChVector3d(0, 0, omega.z))
        
        # Enforce on ice floes
        for floe in self.ice_floes.values():
            if floe.is_bonded:
                for body in floe.particles:
                    pos = body.GetPos()
                    body.SetPos(chrono.ChVector3d(pos.x, pos.y, 0))
                    vel = body.GetPosDt()
                    body.SetPosDt(chrono.ChVector3d(vel.x, vel.y, 0))
            elif floe.body:
                pos = floe.body.GetPos()
                floe.body.SetPos(chrono.ChVector3d(pos.x, pos.y, 0))
                vel = floe.body.GetPosDt()
                floe.body.SetPosDt(chrono.ChVector3d(vel.x, vel.y, 0))
        
    def _mock_step(self, dt: float):
        """Simple mock physics step for testing without Chrono."""
        # Simple Euler integration for ship
        if self.ship_body is not None:
            # Position update
            self._ship_state[0] += self._ship_state[3] * dt
            self._ship_state[1] += self._ship_state[4] * dt
            self._ship_state[2] += self._ship_state[5] * dt
            
            # Simple damping
            damping = 0.99
            self._ship_state[3:] *= damping
            
    def _apply_buoyancy_forces(self):
        """Apply buoyancy forces to all floating bodies.
        
        NOTE: Currently disabled for 2D XY simulation.
        Bodies are constrained to Z=0 via _enforce_planar_constraint().
        """
        # Buoyancy disabled for 2D simulation
        return
                    
    def _apply_buoyancy_to_body(self, body, thickness: float):
        """Apply buoyancy force to a single body."""
        pos = body.GetPos()
        
        # Simplified buoyancy: force proportional to submerged volume
        submerged_depth = self.WATER_LEVEL - pos.z
        if submerged_depth <= 0:
            return
            
        # Clamp to body height
        submerged_depth = min(submerged_depth, thickness)
        
        # Buoyancy force
        mass = body.GetMass()
        volume = mass / self.material.density
        submerged_fraction = submerged_depth / thickness
        buoyancy = self.WATER_DENSITY * self.GRAVITY * volume * submerged_fraction
        
        force = chrono.ChVector3d(0, 0, buoyancy)
        body.AccumulateForce(force, body.GetPos(), False)
        
    def _apply_damping(self):
        """Apply hydrodynamic damping to moving bodies (XY only for 2D simulation)."""
        if self._mock_mode:
            return
            
        # Damping coefficients
        linear_damping = 0.5
        angular_damping = 0.3
        
        # Apply to ship
        if self.ship_body:
            vel = self.ship_body.GetPosDt()
            damping_force = chrono.ChVector3d(
                -linear_damping * vel.x,
                -linear_damping * vel.y,
                0  # No Z damping for 2D
            )
            self.ship_body.AccumulateForce(damping_force, self.ship_body.GetPos(), False)
            
            omega = self.ship_body.GetAngVelParent()
            damping_torque = chrono.ChVector3d(
                0, 0,  # Only Z-axis rotation for 2D
                -angular_damping * omega.z
            )
            self.ship_body.AccumulateTorque(damping_torque, False)
            
    def _check_bond_failures(self):
        """Check all bonds for failure conditions."""
        for floe_id, floe in self.ice_floes.items():
            if not floe.is_bonded or floe.assembly is None:
                continue
                
            newly_broken = []
            
            for bond in floe.assembly.bonds:
                if bond.broken:
                    continue
                    
                if bond.chrono_link is not None and not self._mock_mode:
                    link = bond.chrono_link
                    link_type = getattr(bond, 'link_type', 'rigid')
                    
                    body_a = floe.particles[bond.i]
                    body_b = floe.particles[bond.j]
                    
                    pos_a = np.array([body_a.GetPos().x, body_a.GetPos().y])
                    pos_b = np.array([body_b.GetPos().x, body_b.GetPos().y])
                    
                    bond_dir = pos_b - pos_a
                    bond_length = np.linalg.norm(bond_dir)
                    if bond_length > 0:
                        bond_dir /= bond_length
                    
                    if link_type == 'spring':
                        # TSDA: get spring force directly
                        normal_force = link.GetForce()
                    else:
                        # ChLinkDistance: get reaction force from constraint
                        # The reaction gives the force needed to maintain the constraint
                        reaction = link.GetReaction2()
                        force_vec = np.array([reaction.force.x, reaction.force.y])
                        # Project onto bond axis for normal force
                        normal_force = np.dot(force_vec, bond_dir)
                    
                    # Estimate shear from relative transverse motion
                    vel_a = np.array([body_a.GetPosDt().x, body_a.GetPosDt().y])
                    vel_b = np.array([body_b.GetPosDt().x, body_b.GetPosDt().y])
                    rel_vel = vel_b - vel_a
                    tangent_vel = rel_vel - np.dot(rel_vel, bond_dir) * bond_dir
                    shear_rate = np.linalg.norm(tangent_vel)
                    shear_force = bond.k_s * shear_rate * self.dt * 0.1  # Scale down
                    
                    # Check for failure using Mohr-Coulomb criterion
                    if bond.check_failure(abs(normal_force), shear_force):
                        bond.break_time = self.time
                        newly_broken.append(bond)
                        
                        # Remove Chrono link
                        self.system.Remove(link)
                        bond.chrono_link = None
                        
            # Record fracture events
            if newly_broken:
                floe.assembly.mark_fragments_dirty()
                
                for bond in newly_broken:
                    pi = floe.assembly.particles[bond.i]
                    pj = floe.assembly.particles[bond.j]
                    midpoint = (pi.pos + pj.pos) / 2
                    
                    self.fracture_events.append({
                        'time': self.time,
                        'floe_id': floe_id,
                        'bond_id': bond.id,
                        'break_mode': bond.break_mode,
                        'x': midpoint[0],
                        'y': midpoint[1],
                        'failure_stress': bond.failure_stress,
                        'n_fragments': floe.n_fragments,
                    })
                    
    def _update_ship_state(self):
        """Update internal ship state vector."""
        if self._mock_mode or self.ship_body is None:
            return
            
        pos = self.ship_body.GetPos()
        vel = self.ship_body.GetPosDt()
        rot = self.ship_body.GetRot()
        omega = self.ship_body.GetAngVelParent()  # Angular velocity in parent (world) frame
        
        # Get heading from quaternion
        heading = rot.GetCardanAnglesZYX().z
        
        self._ship_state = np.array([
            pos.x, pos.y, heading,
            vel.x, vel.y, omega.z
        ])
        
    def get_ship_state(self) -> dict:
        """
        Get current ship state.
        
        Returns:
            Dictionary with x, y, heading, vx, vy, omega
        """
        return {
            'x': self._ship_state[0],
            'y': self._ship_state[1],
            'heading': self._ship_state[2],
            'vx': self._ship_state[3],
            'vy': self._ship_state[4],
            'omega': self._ship_state[5],
        }
    
    def get_ship_state_vector(self) -> np.ndarray:
        """Get ship state as [x, y, psi, u, v, r] array."""
        return self._ship_state.copy()
    
    def get_floe_state(self, floe_id: int) -> Optional[dict]:
        """Get state of a specific ice floe."""
        if floe_id not in self.ice_floes:
            return None
            
        floe = self.ice_floes[floe_id]
        
        if self._mock_mode:
            return {
                'id': floe_id,
                'type': floe.floe_type.value,
                'n_particles': floe.n_particles,
                'n_broken_bonds': floe.n_broken_bonds,
                'n_fragments': floe.n_fragments,
            }
            
        if floe.is_bonded:
            # Get centroid from particles
            positions = [np.array([b.GetPos().x, b.GetPos().y]) 
                        for b in floe.particles]
            centroid = np.mean(positions, axis=0) if positions else np.zeros(2)
            
            return {
                'id': floe_id,
                'type': 'bonded',
                'centroid': centroid,
                'n_particles': len(floe.particles),
                'n_broken_bonds': floe.n_broken_bonds,
                'n_fragments': floe.n_fragments,
            }
        else:
            if floe.body:
                pos = floe.body.GetPos()
                vel = floe.body.GetPosDt()
                return {
                    'id': floe_id,
                    'type': 'rigid',
                    'x': pos.x,
                    'y': pos.y,
                    'vx': vel.x,
                    'vy': vel.y,
                }
        return None
    
    def get_total_momentum(self) -> np.ndarray:
        """Get total linear momentum of system."""
        momentum = np.zeros(2)
        
        if self._mock_mode:
            if self.ship_body:
                mass = self.ship_body.get('mass', 10000.0)
                momentum += mass * self._ship_state[3:5]
            return momentum
            
        # Ship momentum
        if self.ship_body:
            mass = self.ship_body.GetMass()
            vel = self.ship_body.GetPosDt()
            momentum += mass * np.array([vel.x, vel.y])
            
        # Ice floe momentum
        for floe in self.ice_floes.values():
            if floe.is_bonded:
                for body in floe.particles:
                    mass = body.GetMass()
                    vel = body.GetPosDt()
                    momentum += mass * np.array([vel.x, vel.y])
            elif floe.body:
                mass = floe.body.GetMass()
                vel = floe.body.GetPosDt()
                momentum += mass * np.array([vel.x, vel.y])
                
        return momentum
    
    def get_total_kinetic_energy(self) -> float:
        """Get total kinetic energy of system."""
        ke = 0.0
        
        if self._mock_mode:
            if self.ship_body:
                mass = self.ship_body.get('mass', 10000.0)
                speed_sq = self._ship_state[3]**2 + self._ship_state[4]**2
                ke += 0.5 * mass * speed_sq
            return ke
            
        # Ship KE
        if self.ship_body:
            mass = self.ship_body.GetMass()
            vel = self.ship_body.GetPosDt()
            speed_sq = vel.x**2 + vel.y**2 + vel.z**2
            ke += 0.5 * mass * speed_sq
            
        # Ice KE
        for floe in self.ice_floes.values():
            if floe.is_bonded:
                for body in floe.particles:
                    mass = body.GetMass()
                    vel = body.GetPosDt()
                    speed_sq = vel.x**2 + vel.y**2 + vel.z**2
                    ke += 0.5 * mass * speed_sq
            elif floe.body:
                mass = floe.body.GetMass()
                vel = floe.body.GetPosDt()
                speed_sq = vel.x**2 + vel.y**2 + vel.z**2
                ke += 0.5 * mass * speed_sq
                
        return ke
    
    def get_fracture_events(self) -> list[dict]:
        """Get list of fracture events since last call."""
        events = self.fracture_events.copy()
        self.fracture_events.clear()
        return events
    
    def debug_collision_status(self):
        """Print collision detection status for debugging."""
        if self._mock_mode:
            print("Mock mode - no real collision detection")
            return
        
        print("=== Collision Debug Status ===")
        
        # Ship collision status
        if self.ship_body:
            enabled = self.ship_body.IsCollisionEnabled()
            model = self.ship_body.GetCollisionModel()
            n_shapes = model.GetNumShapes() if model else 0
            family = model.GetFamily() if model else -1
            pos = self.ship_body.GetPos()
            print(f"Ship: enabled={enabled}, shapes={n_shapes}, family={family}, pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
        else:
            print("Ship: not created")
        
        # Ice floe collision status
        n_rigid = 0
        n_bonded = 0
        for fid, floe in self.ice_floes.items():
            if floe.is_bonded:
                n_bonded += 1
                if floe.particles:
                    body = floe.particles[0]
                    enabled = body.IsCollisionEnabled()
                    model = body.GetCollisionModel()
                    n_shapes = model.GetNumShapes() if model else 0
                    family = model.GetFamily() if model else -1
                    pos = body.GetPos()
                    print(f"Bonded floe {fid} (first particle): enabled={enabled}, shapes={n_shapes}, family={family}, pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
            elif floe.body:
                n_rigid += 1
                enabled = floe.body.IsCollisionEnabled()
                model = floe.body.GetCollisionModel()
                n_shapes = model.GetNumShapes() if model else 0
                family = model.GetFamily() if model else -1
                pos = floe.body.GetPos()
                print(f"Rigid floe {fid}: enabled={enabled}, shapes={n_shapes}, family={family}, pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
        
        print(f"\nTotal: {n_rigid} rigid floes, {n_bonded} bonded floes")
        
        # Contact count
        n_contacts = self.system.GetContactContainer().GetNumContacts()
        print(f"Active contacts: {n_contacts}")
        print("==============================")
    
    def cleanup(self):
        """Clean up Chrono resources."""
        # Clear system if not in mock mode
        if not self._mock_mode and self.system:
            self.system.Clear()
            
        # Always clear internal state
        self.ice_floes.clear()
        self.ship_body = None
        self._floe_counter = 0
