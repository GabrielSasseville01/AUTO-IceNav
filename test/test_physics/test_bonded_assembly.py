"""
Unit tests for bonded particle assembly.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import BondedAssembly, Bond, Particle, IceMaterialParams
from ship_ice_planner.physics.bonded_assembly import crosses_midline, create_test_assembly


class TestParticle:
    """Tests for Particle dataclass."""
    
    def test_particle_creation(self):
        """Particle should be created with position."""
        p = Particle(id=0, pos=np.array([5.0, 5.0]))
        assert p.id == 0
        assert np.allclose(p.pos, [5.0, 5.0])
        
    def test_particle_default_velocity(self):
        """Particle should have zero velocity by default."""
        p = Particle(id=0, pos=np.array([0, 0]))
        assert np.allclose(p.vel, [0, 0])
        
    def test_particle_converts_to_numpy(self):
        """Particle should convert lists to numpy arrays."""
        p = Particle(id=0, pos=[1, 2], vel=[3, 4])
        assert isinstance(p.pos, np.ndarray)
        assert isinstance(p.vel, np.ndarray)


class TestBond:
    """Tests for Bond class."""
    
    def test_bond_creation(self):
        """Bond should be created with particle indices."""
        bond = Bond(id=0, i=0, j=1, rest_length=1.0)
        assert bond.i == 0
        assert bond.j == 1
        assert not bond.broken
        
    def test_bond_tensile_failure(self):
        """Bond should break when tensile force exceeds threshold."""
        bond = Bond(id=0, i=0, j=1, rest_length=1.0, sigma_c=1e5, tau_c=8e4)
        
        # Force below threshold
        bond.check_failure(normal_force=0.5e5, shear_force=0)
        assert not bond.broken
        
        # Force above threshold
        bond.check_failure(normal_force=1.5e5, shear_force=0)
        assert bond.broken
        assert bond.break_mode == 'tensile'
        
    def test_bond_shear_failure(self):
        """Bond should break when shear force exceeds threshold."""
        bond = Bond(id=0, i=0, j=1, rest_length=1.0, sigma_c=1e5, tau_c=8e4)
        
        # Force above shear threshold
        bond.check_failure(normal_force=0, shear_force=1e5)
        assert bond.broken
        assert bond.break_mode == 'shear'
        
    def test_bond_mixed_mode_failure(self):
        """Bond should fail in mixed mode when combined stresses are high."""
        bond = Bond(id=0, i=0, j=1, rest_length=1.0, sigma_c=1e5, tau_c=8e4)
        
        # Combined loading (sigma/sigma_c + tau/tau_c > 1)
        # 0.6 * sigma_c + 0.6 * tau_c should exceed threshold
        bond.check_failure(normal_force=0.6e5, shear_force=0.5e5)
        assert bond.broken
        assert bond.break_mode == 'mixed'
        
    def test_broken_bond_stays_broken(self):
        """Once broken, bond should stay broken."""
        bond = Bond(id=0, i=0, j=1, rest_length=1.0, sigma_c=1e5, tau_c=8e4)
        
        # Break it
        bond.check_failure(normal_force=2e5, shear_force=0)
        assert bond.broken
        
        # Check again with lower force
        bond.check_failure(normal_force=0, shear_force=0)
        assert bond.broken  # Still broken
        
    def test_bond_area_property(self):
        """Bond area should be computed from rest length."""
        bond = Bond(id=0, i=0, j=1, rest_length=2.0)
        assert bond.area > 0


class TestBondedAssemblyCreation:
    """Tests for assembly creation."""
    
    def test_assembly_from_square(self, square_vertices):
        """Assembly should be created from square vertices."""
        assembly = BondedAssembly(square_vertices)
        assert assembly.n_particles > 0
        assert assembly.n_bonds > 0
        
    def test_assembly_from_large_polygon(self, large_polygon):
        """Larger polygon with same particle radius should have more particles."""
        # Use same absolute particle radius for both to compare particle counts
        small = BondedAssembly(np.array([[0,0], [10,0], [10,10], [0,10]]), particle_radius=1.0)
        large = BondedAssembly(large_polygon, particle_radius=1.0)
        assert large.n_particles > small.n_particles
        
    def test_assembly_with_custom_radius(self, square_vertices):
        """Custom particle radius should be respected."""
        assembly = BondedAssembly(square_vertices, particle_radius=0.5)
        assert assembly.particle_radius == 0.5
        
    def test_assembly_with_material(self, square_vertices):
        """Assembly should use provided material."""
        material = IceMaterialParams(density=900.0)
        assembly = BondedAssembly(square_vertices, material=material)
        assert assembly.material.density == 900.0


class TestHexagonalPacking:
    """Tests for hexagonal particle packing."""
    
    def test_particles_inside_polygon(self, square_vertices):
        """All particles should be inside the polygon."""
        from shapely.geometry import Point
        
        assembly = BondedAssembly(square_vertices)
        
        for particle in assembly.particles:
            point = Point(particle.pos)
            # Small buffer for edge particles
            assert assembly.polygon.buffer(assembly.particle_radius * 0.5).contains(point)
            
    def test_coverage_ratio_reasonable(self, square_vertices):
        """Coverage ratio should be close to hex packing optimal."""
        assembly = BondedAssembly(square_vertices, particle_radius=0.5)
        coverage = assembly.coverage_ratio()
        
        # Hex packing optimal is ~0.9069, but with boundary effects expect less
        assert 0.5 < coverage < 1.0
        
    def test_particles_non_overlapping(self, square_vertices):
        """Particles should not significantly overlap."""
        assembly = BondedAssembly(square_vertices)
        
        for i, pi in enumerate(assembly.particles):
            for j, pj in enumerate(assembly.particles):
                if i >= j:
                    continue
                dist = np.linalg.norm(pi.pos - pj.pos)
                # Allow small overlap (touching is OK)
                min_dist = 1.8 * assembly.particle_radius
                assert dist >= min_dist, f"Particles {i} and {j} overlap"


class TestBondCreation:
    """Tests for bond creation between particles."""
    
    def test_bonds_connect_neighbors(self, square_vertices):
        """Bonds should only connect nearby particles."""
        assembly = BondedAssembly(square_vertices)
        
        for bond in assembly.bonds:
            pi = assembly.particles[bond.i]
            pj = assembly.particles[bond.j]
            dist = np.linalg.norm(pi.pos - pj.pos)
            
            # Neighbors should be within ~2.2 radii for hex packing
            max_bond_dist = 2.2 * assembly.particle_radius
            assert dist <= max_bond_dist
            
    def test_interior_particles_have_six_neighbors(self, large_polygon):
        """Interior particles in hex packing should have 6 bonds."""
        assembly = BondedAssembly(large_polygon, particle_radius=1.0)
        
        # Count bonds per particle
        bond_counts = [0] * len(assembly.particles)
        for bond in assembly.bonds:
            bond_counts[bond.i] += 1
            bond_counts[bond.j] += 1
            
        # Interior particles should have 6 neighbors
        # Some will have fewer (boundary) but max should be 6
        assert max(bond_counts) <= 6
        # At least some should have 6
        assert 6 in bond_counts or max(bond_counts) >= 5


class TestFragmentation:
    """Tests for fragment detection after bond breaking."""
    
    def test_initially_one_fragment(self, square_vertices):
        """Initially all particles should be in one fragment."""
        assembly = BondedAssembly(square_vertices)
        fragments = assembly.get_fragments()
        assert len(fragments) == 1
        
    def test_breaking_bonds_creates_fragments(self, square_vertices):
        """Breaking bonds should create multiple fragments."""
        assembly = BondedAssembly(square_vertices, particle_radius=0.5)
        
        # Break bonds crossing the midline
        for bond in assembly.bonds:
            if crosses_midline(bond, assembly, axis='x'):
                bond.broken = True
                
        assembly.mark_fragments_dirty()
        fragments = assembly.get_fragments()
        
        # Should have at least 2 fragments
        assert len(fragments) >= 2
        
    def test_fragment_particles_sum_correct(self, square_vertices):
        """All particles should be in exactly one fragment."""
        assembly = BondedAssembly(square_vertices)
        
        # Break some bonds
        for bond in assembly.bonds[:len(assembly.bonds)//4]:
            bond.broken = True
            
        assembly.mark_fragments_dirty()
        fragments = assembly.get_fragments()
        
        # Count total particles across fragments
        total = sum(len(frag) for frag in fragments)
        assert total == len(assembly.particles)
        
        # Each particle should appear exactly once
        all_indices = []
        for frag in fragments:
            all_indices.extend(frag)
        assert len(set(all_indices)) == len(all_indices)


class TestStressCalculation:
    """Tests for particle stress calculations."""
    
    def test_particle_stress_initially_zero(self, square_vertices):
        """Initially particles should have zero stress."""
        assembly = BondedAssembly(square_vertices)
        
        stress = assembly.get_particle_stress(0)
        assert stress == 0.0
        
    def test_stress_array_correct_length(self, square_vertices):
        """Stress array should have entry for each particle."""
        assembly = BondedAssembly(square_vertices)
        stresses = assembly.get_all_particle_stresses()
        assert len(stresses) == len(assembly.particles)


class TestProperties:
    """Tests for assembly properties."""
    
    def test_total_mass_positive(self, square_vertices):
        """Total mass should be positive."""
        assembly = BondedAssembly(square_vertices)
        assert assembly.total_mass > 0
        
    def test_centroid_inside_polygon(self, square_vertices):
        """Centroid should be inside the polygon."""
        from shapely.geometry import Point
        
        assembly = BondedAssembly(square_vertices)
        centroid = assembly.centroid
        
        assert assembly.polygon.contains(Point(centroid))
        
    def test_intact_bonds_plus_broken_equals_total(self, square_vertices):
        """Intact + broken should equal total bonds."""
        assembly = BondedAssembly(square_vertices)
        
        # Break some bonds
        for bond in assembly.bonds[:5]:
            bond.broken = True
            
        assert assembly.n_intact_bonds + assembly.n_broken_bonds == assembly.n_bonds


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_test_assembly(self):
        """create_test_assembly should produce valid assembly."""
        assembly = create_test_assembly()
        assert assembly.n_particles > 0
        assert assembly.n_bonds > 0
        
    def test_crosses_midline_x_axis(self, square_vertices):
        """crosses_midline should detect bonds crossing center."""
        assembly = BondedAssembly(square_vertices, particle_radius=0.5)
        
        crossing_count = 0
        for bond in assembly.bonds:
            if crosses_midline(bond, assembly, axis='x'):
                crossing_count += 1
                
        # Should have some crossing bonds
        assert crossing_count > 0
        # But not all bonds should cross
        assert crossing_count < len(assembly.bonds)


class TestRepr:
    """Tests for string representation."""
    
    def test_repr_shows_counts(self, square_vertices):
        """repr should show particle and bond counts."""
        assembly = BondedAssembly(square_vertices)
        s = repr(assembly)
        
        assert 'particles=' in s
        assert 'bonds=' in s
