"""
Ice material parameters for bonded DEM simulation.

This module defines the physical properties of sea ice used in the
PyChrono-based bonded DEM simulation. Default values are based on
published ice mechanics literature and ISO 19906 standards.

The bond failure criterion follows Mohr-Coulomb as described in:
  Celigueta et al. (2019) "A local bonded DEM approach for ice modeling"
  Computational Particle Mechanics, 6:739-765
  
Key equations from paper:
  - Tensile failure (Eq. 18): Fnt >= σ_t × A
  - Shear failure (Eq. 19): Fs >= τ_f × A + μ × |Fn_compression|
  
References:
- Celigueta et al. (2019) - Bonded DEM for ice fracture
- ISO 19906:2019 - Arctic offshore structures
- Timco & Weeks (2010) - A review of the engineering properties of sea ice
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class IceMaterialParams:
    """
    Material parameters for sea ice in bonded DEM simulation.
    
    Default values based on Table 3 from Celigueta et al. (2019):
    - Density: 920 kg/m³
    - Young's modulus: 1 GPa
    - Poisson ratio: 0.33
    - Friction angle: 30° (μ = tan(30°) = 0.577)
    - Tensile strength: 1.6 MPa
    - Shear strength: 1.0 MPa
    
    Attributes:
        density: Ice density in kg/m³ (paper: 920)
        youngs_modulus: Young's modulus in Pa (paper: 1 GPa)
        poisson_ratio: Poisson's ratio (paper: 0.33)
        tensile_strength: Tensile strength in Pa (paper: 1.6 MPa)
        shear_strength: Shear strength in Pa (paper: 1.0 MPa)
        compressive_strength: Compressive strength in Pa
        friction: Friction coefficient μ = tan(φ) for Mohr-Coulomb (paper: tan(30°)=0.577)
        restitution: Coefficient of restitution for collisions
        thickness: Default ice thickness in meters
    """
    
    # Density (kg/m³) - Paper Table 3: 920
    density: float = 920.0
    
    # Elastic properties - Paper Table 3: E=1 GPa, ν=0.33
    youngs_modulus: float = 1.0e9  # Pa (1 GPa - from paper)
    poisson_ratio: float = 0.33
    
    # Strength properties (bond breaking thresholds)
    # Paper Table 3: σ_t = 1.6 MPa, τ_f = 1.0 MPa
    tensile_strength: float = 1.6e6   # Pa (1.6 MPa - from paper)
    shear_strength: float = 1.0e6     # Pa (1.0 MPa - from paper)
    compressive_strength: float = 5.0e6  # Pa (5 MPa)
    
    # Contact/friction properties
    # Paper Table 3: friction angle φ = 30°, so μ = tan(30°) = 0.577
    friction: float = 0.577  # Mohr-Coulomb friction coefficient
    restitution: float = 0.05  # Low for brittle behavior (paper recommends 0.05)
    
    # Geometric defaults
    thickness: float = 1.0  # meters
    
    # Derived properties (computed in __post_init__)
    k_n: float = field(init=False)  # Normal contact stiffness
    k_s: float = field(init=False)  # Shear contact stiffness
    
    def __post_init__(self):
        """Compute derived stiffness values from elastic properties."""
        # Normal stiffness from Young's modulus
        self.k_n = self.youngs_modulus
        
        # Shear stiffness from shear modulus G = E / (2 * (1 + nu))
        shear_modulus = self.youngs_modulus / (2 * (1 + self.poisson_ratio))
        self.k_s = shear_modulus
    
    def get_bond_stiffness(self, particle_radius: float, 
                            target_failure_strain: float = 0.05) -> tuple[float, float]:
        """
        Calculate bond stiffness for DEM simulation.
        
        Instead of using true material stiffness (which causes instant fracture
        due to numerical noise), we calibrate stiffness so that bonds break
        at a target strain level (default 5%).
        
        This follows the approach in Celigueta et al. (2019) where bond parameters
        are calibrated for numerical stability while preserving failure behavior.
        
        Args:
            particle_radius: Radius of bonded particles in meters
            target_failure_strain: Strain at which bond should fail (default 5%)
            
        Returns:
            Tuple of (normal_stiffness, shear_stiffness) in N/m
        """
        # Bond area for force calculation
        bond_area = np.pi * particle_radius ** 2
        bond_length = 2 * particle_radius
        
        # Breaking force
        sigma_c = self.tensile_strength * bond_area  # Tensile limit (N)
        
        # Target deformation at failure
        target_deform = bond_length * target_failure_strain
        
        # Stiffness = Force / Deformation at failure
        k_n_bond = sigma_c / target_deform
        
        # Shear stiffness typically ~0.5-0.8 of normal
        k_s_bond = k_n_bond * 0.6
        
        return k_n_bond, k_s_bond
    
    def get_bond_strength(self, particle_radius: float) -> tuple[float, float]:
        """
        Calculate bond breaking forces for a given particle size.
        
        Args:
            particle_radius: Radius of bonded particles in meters
            
        Returns:
            Tuple of (tensile_force_limit, shear_force_limit) in N
        """
        # Bond area for force calculation
        bond_area = np.pi * particle_radius ** 2
        
        tensile_limit = self.tensile_strength * bond_area
        shear_limit = self.shear_strength * bond_area
        
        return tensile_limit, shear_limit
    
    def get_damping_coefficients(self, particle_mass: float, 
                                  critical_damping_ratio: float = 0.1) -> tuple[float, float]:
        """
        Calculate damping coefficients for contact model.
        
        Uses critical damping ratio approach for stable simulation.
        
        Args:
            particle_mass: Mass of a single particle in kg
            critical_damping_ratio: Fraction of critical damping (0-1)
            
        Returns:
            Tuple of (normal_damping, tangential_damping) coefficients
        """
        # Critical damping: c_crit = 2 * sqrt(k * m)
        c_n_crit = 2 * np.sqrt(self.k_n * particle_mass)
        c_s_crit = 2 * np.sqrt(self.k_s * particle_mass)
        
        return c_n_crit * critical_damping_ratio, c_s_crit * critical_damping_ratio
    
    @classmethod
    def first_year_ice(cls, temperature: float = -10.0) -> 'IceMaterialParams':
        """
        Create parameters for first-year sea ice.
        
        Args:
            temperature: Ice temperature in Celsius (affects strength)
            
        Returns:
            IceMaterialParams configured for first-year ice
        """
        # Temperature-dependent strength modification
        # Strength increases roughly 0.05 MPa per degree below 0°C
        temp_factor = 1.0 + 0.1 * abs(temperature) / 10.0
        
        return cls(
            density=910.0,
            youngs_modulus=4.0e9,
            tensile_strength=0.4e6 * temp_factor,
            shear_strength=0.35e6 * temp_factor,
            compressive_strength=4.0e6 * temp_factor,
        )
    
    @classmethod
    def multi_year_ice(cls) -> 'IceMaterialParams':
        """
        Create parameters for multi-year sea ice.
        
        Multi-year ice is stronger and denser than first-year ice.
        
        Returns:
            IceMaterialParams configured for multi-year ice
        """
        return cls(
            density=880.0,  # Lower due to freshwater content
            youngs_modulus=8.0e9,
            tensile_strength=0.8e6,
            shear_strength=0.7e6,
            compressive_strength=8.0e6,
            thickness=2.0,  # Typically thicker
        )
    
    @classmethod
    def from_config(cls, config: dict) -> 'IceMaterialParams':
        """
        Create parameters from configuration dictionary.
        
        Args:
            config: Dictionary with ice_material configuration
            
        Returns:
            IceMaterialParams instance
        """
        ice_cfg = config.get('ice_material', {})
        return cls(
            density=float(ice_cfg.get('density', 920.0)),
            youngs_modulus=float(ice_cfg.get('youngs_modulus', 5.0e9)),
            poisson_ratio=float(ice_cfg.get('poisson_ratio', 0.33)),
            tensile_strength=float(ice_cfg.get('tensile_strength', 0.5e6)),
            shear_strength=float(ice_cfg.get('shear_strength', 0.4e6)),
            compressive_strength=float(ice_cfg.get('compressive_strength', 5.0e6)),
            friction=float(ice_cfg.get('friction', 0.1)),
            restitution=float(ice_cfg.get('restitution', 0.3)),
            thickness=float(ice_cfg.get('thickness', 1.0)),
        )
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary for serialization."""
        return {
            'density': self.density,
            'youngs_modulus': self.youngs_modulus,
            'poisson_ratio': self.poisson_ratio,
            'tensile_strength': self.tensile_strength,
            'shear_strength': self.shear_strength,
            'compressive_strength': self.compressive_strength,
            'friction': self.friction,
            'restitution': self.restitution,
            'thickness': self.thickness,
        }
    
    def __repr__(self) -> str:
        return (
            f"IceMaterialParams(density={self.density} kg/m³, "
            f"E={self.youngs_modulus/1e9:.1f} GPa, "
            f"σ_t={self.tensile_strength/1e6:.2f} MPa, "
            f"τ={self.shear_strength/1e6:.2f} MPa)"
        )
