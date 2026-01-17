"""
Unit tests for ice material parameters.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import IceMaterialParams


class TestIceMaterialDefaults:
    """Tests for default material parameter values."""
    
    def test_material_defaults_exist(self):
        """Verify default parameters are set."""
        mat = IceMaterialParams()
        assert mat.density is not None
        assert mat.youngs_modulus is not None
        assert mat.tensile_strength is not None
        assert mat.shear_strength is not None
        
    def test_density_in_valid_range(self):
        """Verify default density is realistic for sea ice."""
        mat = IceMaterialParams()
        # Sea ice density typically 900-940 kg/m³
        assert 900 <= mat.density <= 940
        
    def test_youngs_modulus_in_valid_range(self):
        """Verify Young's modulus is realistic."""
        mat = IceMaterialParams()
        # Sea ice E typically 1-10 GPa
        assert 1e9 <= mat.youngs_modulus <= 10e9
        
    def test_tensile_strength_in_valid_range(self):
        """Verify tensile strength is realistic."""
        mat = IceMaterialParams()
        # Sea ice tensile strength typically 0.2-2 MPa
        assert 0.1e6 <= mat.tensile_strength <= 2e6
        
    def test_shear_strength_less_than_tensile(self):
        """Shear strength should generally be less than tensile."""
        mat = IceMaterialParams()
        assert mat.shear_strength <= mat.tensile_strength


class TestDerivedProperties:
    """Tests for computed/derived material properties."""
    
    def test_stiffness_computed_in_post_init(self):
        """Normal and shear stiffness should be computed."""
        mat = IceMaterialParams(youngs_modulus=5e9, poisson_ratio=0.33)
        assert mat.k_n > 0
        assert mat.k_s > 0
        
    def test_normal_stiffness_equals_youngs_modulus(self):
        """k_n should equal Young's modulus."""
        E = 5e9
        mat = IceMaterialParams(youngs_modulus=E)
        assert mat.k_n == E
        
    def test_shear_stiffness_from_shear_modulus(self):
        """k_s should be derived from shear modulus G = E/(2*(1+nu))."""
        E = 5e9
        nu = 0.33
        mat = IceMaterialParams(youngs_modulus=E, poisson_ratio=nu)
        
        expected_G = E / (2 * (1 + nu))
        assert abs(mat.k_s - expected_G) < 1e6  # Within 1 MPa


class TestBondParameters:
    """Tests for bond stiffness and strength calculations."""
    
    def test_bond_stiffness_positive(self):
        """Bond stiffness should be positive."""
        mat = IceMaterialParams()
        k_n, k_s = mat.get_bond_stiffness(particle_radius=0.5)
        assert k_n > 0
        assert k_s > 0
        
    def test_bond_stiffness_scales_with_radius(self):
        """Larger particles should have stiffer bonds."""
        mat = IceMaterialParams()
        k_n_small, _ = mat.get_bond_stiffness(particle_radius=0.5)
        k_n_large, _ = mat.get_bond_stiffness(particle_radius=1.0)
        # Stiffness scales with area/length ~ r
        assert k_n_large > k_n_small
        
    def test_bond_strength_positive(self):
        """Bond strength should be positive."""
        mat = IceMaterialParams()
        sigma_c, tau_c = mat.get_bond_strength(particle_radius=0.5)
        assert sigma_c > 0
        assert tau_c > 0
        
    def test_bond_strength_scales_with_area(self):
        """Larger bonds should be stronger (force scales with area)."""
        mat = IceMaterialParams()
        sigma_small, _ = mat.get_bond_strength(particle_radius=0.5)
        sigma_large, _ = mat.get_bond_strength(particle_radius=1.0)
        # Strength (force) scales with area ~ r²
        assert sigma_large > sigma_small * 3  # Should be ~4x larger


class TestDampingCoefficients:
    """Tests for damping coefficient calculations."""
    
    def test_damping_coefficients_positive(self):
        """Damping coefficients should be positive."""
        mat = IceMaterialParams()
        c_n, c_s = mat.get_damping_coefficients(particle_mass=100.0)
        assert c_n > 0
        assert c_s > 0
        
    def test_critical_damping_ratio_affects_damping(self):
        """Higher critical damping ratio should give more damping."""
        mat = IceMaterialParams()
        c_low, _ = mat.get_damping_coefficients(particle_mass=100.0, critical_damping_ratio=0.1)
        c_high, _ = mat.get_damping_coefficients(particle_mass=100.0, critical_damping_ratio=0.5)
        assert c_high > c_low


class TestFactoryMethods:
    """Tests for factory methods creating specific ice types."""
    
    def test_first_year_ice_creation(self):
        """First-year ice should be created successfully."""
        mat = IceMaterialParams.first_year_ice()
        assert mat.density > 0
        assert mat.tensile_strength > 0
        
    def test_first_year_ice_temperature_effect(self):
        """Colder temperature should increase strength."""
        mat_warm = IceMaterialParams.first_year_ice(temperature=-5.0)
        mat_cold = IceMaterialParams.first_year_ice(temperature=-20.0)
        assert mat_cold.tensile_strength > mat_warm.tensile_strength
        
    def test_multi_year_ice_stronger_than_first_year(self):
        """Multi-year ice should be stronger."""
        fy = IceMaterialParams.first_year_ice()
        my = IceMaterialParams.multi_year_ice()
        assert my.tensile_strength > fy.tensile_strength
        assert my.youngs_modulus > fy.youngs_modulus
        
    def test_from_config_creates_valid_material(self):
        """Material should be created from config dictionary."""
        config = {
            'ice_material': {
                'density': 900.0,
                'tensile_strength': 1.0e6,
            }
        }
        mat = IceMaterialParams.from_config(config)
        assert mat.density == 900.0
        assert mat.tensile_strength == 1.0e6


class TestSerialization:
    """Tests for serialization/deserialization."""
    
    def test_to_dict_contains_all_properties(self):
        """to_dict should include all main properties."""
        mat = IceMaterialParams()
        d = mat.to_dict()
        
        assert 'density' in d
        assert 'youngs_modulus' in d
        assert 'tensile_strength' in d
        assert 'shear_strength' in d
        assert 'friction' in d
        
    def test_roundtrip_through_dict(self):
        """Material should survive roundtrip through dict."""
        original = IceMaterialParams(density=905.0, tensile_strength=0.6e6)
        d = original.to_dict()
        
        # Create new material from dict
        config = {'ice_material': d}
        restored = IceMaterialParams.from_config(config)
        
        assert restored.density == original.density
        assert restored.tensile_strength == original.tensile_strength


class TestRepr:
    """Tests for string representation."""
    
    def test_repr_contains_key_values(self):
        """repr should show key material values."""
        mat = IceMaterialParams()
        s = repr(mat)
        
        assert 'density' in s.lower() or 'kg/m³' in s
        assert 'GPa' in s or 'Pa' in s
