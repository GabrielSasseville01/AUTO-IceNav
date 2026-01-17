"""
Bonded particle assembly for DEM ice floe simulation.

This module creates hexagonally-packed particle assemblies from polygon
shapes and manages breakable bonds between particles. When bonds break,
the assembly can fracture into multiple fragments.

The bonded DEM approach allows realistic simulation of:
- Ice fracture under ship impact
- Crack propagation through ice floes
- Fragment generation and separation
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from collections import deque

from .ice_material import IceMaterialParams


@dataclass
class Particle:
    """
    A single DEM particle in the bonded assembly.
    
    Attributes:
        id: Unique particle identifier
        pos: Position as numpy array [x, y]
        vel: Velocity as numpy array [vx, vy]
        radius: Particle radius in meters
        mass: Particle mass in kg
        force: Accumulated force vector
        angular_vel: Angular velocity in rad/s
        torque: Accumulated torque
    """
    id: int
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0
    mass: float = 1.0
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angular_vel: float = 0.0
    torque: float = 0.0
    
    # Reference to Chrono body (set when added to simulator)
    chrono_body: Optional[object] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.pos = np.asarray(self.pos, dtype=float)
        self.vel = np.asarray(self.vel, dtype=float)
        self.force = np.asarray(self.force, dtype=float)


@dataclass 
class Bond:
    """
    A breakable bond connecting two particles.
    
    Implements the Mohr-Coulomb failure criterion from:
    Celigueta et al. (2019) "Discrete Element Method for ice modeling"
    Computational Particle Mechanics, 6:739-765
    
    Failure criteria (Eq. 18-19 from paper):
    - Tensile: Fn >= σ_t × A  (normal tensile force exceeds strength)
    - Shear:   Fs >= τ_f × A + μ × |Fn_c|  (Mohr-Coulomb with friction)
    
    Where:
    - Fn = normal force (positive = tension, negative = compression)
    - Fs = shear force magnitude
    - σ_t = tensile strength (Pa)
    - τ_f = shear strength (Pa)
    - μ = friction coefficient = tan(friction_angle)
    - A = bond cross-sectional area
    - Fn_c = compressive normal force (negative of Fn when Fn < 0)
    
    Attributes:
        id: Unique bond identifier
        i: Index of first particle
        j: Index of second particle
        rest_length: Initial bond length (equilibrium distance)
        particle_radius: Radius of bonded particles (for area calculation)
        k_n: Normal stiffness (N/m)
        k_s: Shear stiffness (N/m)
        sigma_c: Tensile strength (N) - force limit
        tau_c: Base shear strength (N) - force limit without friction
        friction_coeff: Friction coefficient μ = tan(φ) for Mohr-Coulomb
        broken: Whether the bond has failed
        break_mode: How the bond failed ('tensile', 'shear', 'mixed')
        failure_stress: Stress at which bond failed
    """
    id: int
    i: int  # First particle index
    j: int  # Second particle index
    rest_length: float
    particle_radius: float = 1.0  # For proper area calculation
    k_n: float = 1e6  # Normal stiffness
    k_s: float = 1e6  # Shear stiffness
    sigma_c: float = 1e5  # Tensile strength (force)
    tau_c: float = 8e4   # Base shear strength (force)
    friction_coeff: float = 0.577  # tan(30°) from paper, friction angle φ=30°
    broken: bool = False
    break_mode: Optional[str] = None
    failure_stress: float = 0.0
    break_time: float = -1.0
    
    # Current force state (updated during simulation)
    normal_force: float = field(default=0.0, repr=False)
    shear_force: float = field(default=0.0, repr=False)
    
    # Reference to Chrono link (set when added to simulator)
    chrono_link: Optional[object] = field(default=None, repr=False)
    
    # Positions for visualization (updated from particles)
    pos_a: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)
    pos_b: np.ndarray = field(default_factory=lambda: np.zeros(2), repr=False)
    
    @property
    def area(self) -> float:
        """
        Bond cross-sectional area.
        
        From paper Eq. 8: A_ij = π × r_c²
        where r_c is the contact radius, approximated as particle_radius.
        """
        return np.pi * self.particle_radius ** 2
    
    def check_failure(self, normal_force: float, shear_force: float) -> bool:
        """
        Check if bond should break under current forces using Mohr-Coulomb criterion.
        
        Implements failure criteria from Celigueta et al. (2019) Eq. 18-19:
        - Tensile: Fnt >= Fnt_limit where Fnt_limit = σ_t × A
        - Shear:   Fs >= Fs_limit where Fs_limit = τ_f × A + μ × |Fn_compression|
        
        The shear strength INCREASES under compression (Mohr-Coulomb effect).
        
        Args:
            normal_force: Normal force (positive = tension, negative = compression)
            shear_force: Magnitude of shear force
            
        Returns:
            True if bond fails, False otherwise
        """
        if self.broken:
            return True
            
        self.normal_force = normal_force
        self.shear_force = shear_force
        
        # 1. Check tensile failure (Eq. 18: Fnt >= Fnt_limit)
        # Only fails in tension (positive normal force)
        if normal_force > self.sigma_c:
            self.broken = True
            self.break_mode = 'tensile'
            self.failure_stress = normal_force / self.area
            return True
        
        # 2. Check shear failure with Mohr-Coulomb criterion (Eq. 19)
        # Fs_limit = τ_f × A + μ × |Fn_compression|
        # When under compression (normal_force < 0), friction adds to shear strength
        compressive_force = max(0.0, -normal_force)  # Only positive compression
        shear_limit = self.tau_c + self.friction_coeff * compressive_force
        
        if abs(shear_force) > shear_limit:
            self.broken = True
            self.break_mode = 'shear'
            self.failure_stress = abs(shear_force) / self.area
            return True
            
        # 3. Mixed mode failure envelope (tension + shear interaction)
        # Linear interaction: (Fn/σ_c) + (Fs/τ_c) > 1
        # Only applies in tension regime
        if normal_force > 0:
            tension_ratio = normal_force / self.sigma_c
            shear_ratio = abs(shear_force) / self.tau_c
            if tension_ratio + shear_ratio > 1.0:
                self.broken = True
                self.break_mode = 'mixed'
                self.failure_stress = max(normal_force, abs(shear_force)) / self.area
                return True
                
        return False
    
    def apply_normal_force(self, force: float):
        """Apply normal force and check for failure."""
        self.check_failure(force, self.shear_force)
        
    def apply_shear_force(self, force: float):
        """Apply shear force and check for failure."""
        self.check_failure(self.normal_force, force)


class BondedAssembly:
    """
    A bonded particle assembly representing an ice floe.
    
    Creates hexagonal particle packing within a polygon boundary and
    connects neighboring particles with breakable bonds.
    
    Attributes:
        vertices: Original polygon vertices
        particles: List of Particle objects
        bonds: List of Bond objects
        material: Ice material parameters
        particle_radius: Radius of each particle
    """
    
    def __init__(self, 
                 vertices: np.ndarray,
                 particle_radius: Optional[float] = None,
                 material: Optional[IceMaterialParams] = None,
                 size_ratio: float = 0.08):
        """
        Create a bonded particle assembly from polygon vertices.
        
        Args:
            vertices: Nx2 array of polygon vertices (CCW order)
            particle_radius: Radius of particles (if None, computed from size_ratio)
            material: Ice material parameters
            size_ratio: Particle radius as fraction of polygon size
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.material = material or IceMaterialParams()
        self.polygon = Polygon(self.vertices)
        
        # Compute particle radius from polygon size if not specified
        if particle_radius is None:
            # Use characteristic length (sqrt of area)
            char_length = np.sqrt(self.polygon.area)
            self.particle_radius = char_length * size_ratio
        else:
            self.particle_radius = particle_radius
            
        # Ensure reasonable particle size
        self.particle_radius = max(self.particle_radius, 0.1)  # Min 10cm
        
        # Create particle packing
        self.particles: list[Particle] = []
        self._create_hex_packing()
        
        # Create bonds between neighbors
        self.bonds: list[Bond] = []
        self._create_bonds()
        
        # Track fracture state
        self._fragments: Optional[list[list[int]]] = None
        self._fragments_dirty = True
    
    def _create_hex_packing(self):
        """Create hexagonal particle packing within polygon."""
        # Get bounding box
        min_x, min_y, max_x, max_y = self.polygon.bounds
        
        # Hex packing parameters
        r = self.particle_radius
        dx = 2 * r  # Horizontal spacing
        dy = r * np.sqrt(3)  # Vertical spacing for hex
        
        # Compute particle mass from material and thickness
        particle_volume = np.pi * r**2 * self.material.thickness
        particle_mass = self.material.density * particle_volume
        
        # Generate grid points
        particle_id = 0
        row = 0
        y = min_y + r
        
        while y < max_y - r:
            # Offset every other row for hex pattern
            x_offset = r if row % 2 else 0
            x = min_x + r + x_offset
            
            while x < max_x - r:
                point = Point(x, y)
                
                # Check if point is inside polygon (with small buffer)
                if self.polygon.contains(point.buffer(r * 0.3)):
                    self.particles.append(Particle(
                        id=particle_id,
                        pos=np.array([x, y]),
                        radius=r,
                        mass=particle_mass,
                    ))
                    particle_id += 1
                    
                x += dx
            y += dy
            row += 1
            
    def _create_bonds(self):
        """Create bonds between neighboring particles."""
        if len(self.particles) < 2:
            return
            
        # Build KD-tree for neighbor search
        positions = np.array([p.pos for p in self.particles])
        tree = cKDTree(positions)
        
        # Find neighbors within bonding distance
        # Hex packing: neighbors at distance 2*r (touching) or 2*r*sqrt(3) (diagonal)
        bond_dist = 2.2 * self.particle_radius
        
        # Get bond stiffness and strength from material
        k_n, k_s = self.material.get_bond_stiffness(self.particle_radius)
        sigma_c, tau_c = self.material.get_bond_strength(self.particle_radius)
        
        # Friction coefficient for Mohr-Coulomb criterion
        # From paper: μ = tan(φ) where φ is friction angle (typically 30° for ice)
        friction_coeff = self.material.friction  # Use material friction
        
        # Find all pairs within bonding distance
        pairs = tree.query_pairs(bond_dist)
        
        bond_id = 0
        for i, j in pairs:
            pi, pj = self.particles[i], self.particles[j]
            rest_length = np.linalg.norm(pi.pos - pj.pos)
            
            self.bonds.append(Bond(
                id=bond_id,
                i=i,
                j=j,
                rest_length=rest_length,
                particle_radius=self.particle_radius,  # For area calculation
                k_n=k_n,
                k_s=k_s,
                sigma_c=sigma_c,
                tau_c=tau_c,
                friction_coeff=friction_coeff,  # Mohr-Coulomb friction
                pos_a=pi.pos.copy(),
                pos_b=pj.pos.copy(),
            ))
            bond_id += 1
            
    def update_bond_positions(self):
        """Update bond endpoint positions from particles."""
        for bond in self.bonds:
            bond.pos_a = self.particles[bond.i].pos.copy()
            bond.pos_b = self.particles[bond.j].pos.copy()
            
    @property
    def n_particles(self) -> int:
        """Number of particles in assembly."""
        return len(self.particles)
    
    @property
    def n_bonds(self) -> int:
        """Number of bonds in assembly."""
        return len(self.bonds)
    
    @property
    def n_intact_bonds(self) -> int:
        """Number of intact (unbroken) bonds."""
        return sum(1 for b in self.bonds if not b.broken)
    
    @property
    def n_broken_bonds(self) -> int:
        """Number of broken bonds."""
        return sum(1 for b in self.bonds if b.broken)
    
    @property
    def total_mass(self) -> float:
        """Total mass of all particles."""
        return sum(p.mass for p in self.particles)
    
    @property
    def centroid(self) -> np.ndarray:
        """Mass-weighted centroid of assembly."""
        if not self.particles:
            return np.array(self.polygon.centroid.coords[0])
        total_mass = self.total_mass
        return sum(p.pos * p.mass for p in self.particles) / total_mass
    
    def get_fragments(self) -> list[list[int]]:
        """
        Get connected components (fragments) after bond breaking.
        
        Uses graph traversal to find groups of particles still
        connected by intact bonds.
        
        Returns:
            List of particle index lists, one per fragment
        """
        if not self._fragments_dirty and self._fragments is not None:
            return self._fragments
            
        # Build adjacency list from intact bonds
        n = len(self.particles)
        adjacency = [[] for _ in range(n)]
        
        for bond in self.bonds:
            if not bond.broken:
                adjacency[bond.i].append(bond.j)
                adjacency[bond.j].append(bond.i)
                
        # Find connected components using BFS
        visited = [False] * n
        fragments = []
        
        for start in range(n):
            if visited[start]:
                continue
                
            # BFS from this particle
            fragment = []
            queue = deque([start])
            visited[start] = True
            
            while queue:
                curr = queue.popleft()
                fragment.append(curr)
                
                for neighbor in adjacency[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        
            fragments.append(fragment)
            
        self._fragments = fragments
        self._fragments_dirty = False
        return fragments
    
    def mark_fragments_dirty(self):
        """Mark that fragments need to be recomputed."""
        self._fragments_dirty = True
        
    def get_fragment_polygons(self) -> list[np.ndarray]:
        """
        Get convex hull polygons for each fragment.
        
        Returns:
            List of Nx2 vertex arrays for each fragment
        """
        from scipy.spatial import ConvexHull
        
        fragments = self.get_fragments()
        polygons = []
        
        for frag_indices in fragments:
            if len(frag_indices) < 3:
                continue
                
            positions = np.array([self.particles[i].pos for i in frag_indices])
            
            try:
                hull = ConvexHull(positions)
                hull_vertices = positions[hull.vertices]
                polygons.append(hull_vertices)
            except Exception:
                # Degenerate case (collinear points)
                continue
                
        return polygons
    
    def get_particle_stress(self, particle_idx: int) -> float:
        """
        Compute cumulative stress on a particle from its bonds.
        
        Args:
            particle_idx: Index of particle
            
        Returns:
            Maximum stress magnitude on particle
        """
        max_stress = 0.0
        
        for bond in self.bonds:
            if bond.i == particle_idx or bond.j == particle_idx:
                # Compute stress from bond forces
                normal_stress = abs(bond.normal_force) / bond.area if bond.area > 0 else 0
                shear_stress = abs(bond.shear_force) / bond.area if bond.area > 0 else 0
                stress = np.sqrt(normal_stress**2 + shear_stress**2)
                max_stress = max(max_stress, stress)
                
        return max_stress
    
    def get_all_particle_stresses(self) -> np.ndarray:
        """Get stress values for all particles."""
        return np.array([self.get_particle_stress(i) for i in range(len(self.particles))])
    
    def coverage_ratio(self) -> float:
        """
        Compute ratio of particle area to polygon area.
        
        Returns:
            Coverage ratio (ideal hex packing ≈ 0.9069)
        """
        particle_area = len(self.particles) * np.pi * self.particle_radius**2
        return particle_area / self.polygon.area
    
    def __repr__(self) -> str:
        return (
            f"BondedAssembly(particles={self.n_particles}, "
            f"bonds={self.n_bonds}, "
            f"broken={self.n_broken_bonds}, "
            f"fragments={len(self.get_fragments())})"
        )


def create_test_assembly(material: Optional[IceMaterialParams] = None,
                          size: float = 20.0) -> BondedAssembly:
    """
    Create a simple square test assembly.
    
    Args:
        material: Ice material parameters
        size: Side length of square in meters
        
    Returns:
        BondedAssembly for testing
    """
    vertices = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ])
    return BondedAssembly(vertices, material=material)


def crosses_midline(bond: Bond, assembly: BondedAssembly, 
                    axis: str = 'x') -> bool:
    """
    Check if a bond crosses the midline of the assembly.
    
    Useful for testing fragmentation.
    
    Args:
        bond: Bond to check
        assembly: Parent assembly
        axis: 'x' or 'y' for midline orientation
        
    Returns:
        True if bond crosses midline
    """
    pi = assembly.particles[bond.i]
    pj = assembly.particles[bond.j]
    
    centroid = assembly.centroid
    
    if axis == 'x':
        mid = centroid[0]
        return (pi.pos[0] < mid) != (pj.pos[0] < mid)
    else:
        mid = centroid[1]
        return (pi.pos[1] < mid) != (pj.pos[1] < mid)
