"""
Unit tests for Chrono physics backend.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import ChronoIceSimulator, IceFloe, IceMaterialParams
from ship_ice_planner.physics.chrono_backend import FloeType


class TestSimulatorCreation:
    """Tests for simulator initialization."""
    
    def test_create_simulator(self):
        """Simulator should be created successfully."""
        sim = ChronoIceSimulator()
        assert sim is not None
        sim.cleanup()
        
    def test_create_with_material(self):
        """Simulator should accept custom material."""
        material = IceMaterialParams(density=900.0)
        sim = ChronoIceSimulator(material=material)
        assert sim.material.density == 900.0
        sim.cleanup()
        
    def test_create_with_config(self, default_config):
        """Simulator should accept configuration dictionary."""
        sim = ChronoIceSimulator(config=default_config)
        assert sim.dt == default_config['physics']['dt']
        sim.cleanup()
        
    def test_initial_time_is_zero(self):
        """Initial simulation time should be zero."""
        sim = ChronoIceSimulator()
        assert sim.time == 0.0
        sim.cleanup()


class TestShipCreation:
    """Tests for ship body creation."""
    
    def test_create_ship(self, chrono_sim):
        """Ship should be created with position and heading."""
        chrono_sim.create_ship(position=(100, 50), heading=np.pi/2)
        
        state = chrono_sim.get_ship_state()
        assert abs(state['x'] - 100) < 0.01
        assert abs(state['y'] - 50) < 0.01
        assert abs(state['heading'] - np.pi/2) < 0.01
        
    def test_ship_initial_velocity_zero(self, chrono_sim):
        """Ship should start with zero velocity."""
        chrono_sim.create_ship(position=(0, 0), heading=0)
        
        state = chrono_sim.get_ship_state()
        assert abs(state['vx']) < 0.01
        assert abs(state['vy']) < 0.01
        assert abs(state['omega']) < 0.01
        
    def test_ship_state_vector_format(self, chrono_sim):
        """State vector should be [x, y, psi, u, v, r]."""
        chrono_sim.create_ship(position=(10, 20), heading=0.5)
        
        vec = chrono_sim.get_ship_state_vector()
        assert vec.shape == (6,)
        assert abs(vec[0] - 10) < 0.01  # x
        assert abs(vec[1] - 20) < 0.01  # y
        assert abs(vec[2] - 0.5) < 0.01  # psi


class TestRigidFloeCreation:
    """Tests for rigid ice floe creation."""
    
    def test_create_rigid_floe(self, chrono_sim, square_vertices):
        """Rigid floe should be created successfully."""
        floe_id = chrono_sim.create_ice_floe_rigid(square_vertices, thickness=1.0)
        
        assert floe_id in chrono_sim.ice_floes
        assert not chrono_sim.ice_floes[floe_id].is_bonded
        
    def test_rigid_floe_properties(self, chrono_sim, square_vertices):
        """Rigid floe should have correct properties."""
        floe_id = chrono_sim.create_ice_floe_rigid(square_vertices, thickness=2.0)
        floe = chrono_sim.ice_floes[floe_id]
        
        assert floe.thickness == 2.0
        assert floe.floe_type == FloeType.RIGID
        
    def test_multiple_rigid_floes(self, chrono_sim, square_vertices):
        """Multiple floes should get unique IDs."""
        id1 = chrono_sim.create_ice_floe_rigid(square_vertices)
        id2 = chrono_sim.create_ice_floe_rigid(square_vertices + [50, 0])
        id3 = chrono_sim.create_ice_floe_rigid(square_vertices + [100, 0])
        
        assert len({id1, id2, id3}) == 3
        assert len(chrono_sim.ice_floes) == 3


class TestBondedFloeCreation:
    """Tests for bonded DEM ice floe creation."""
    
    def test_create_bonded_floe(self, chrono_sim, large_polygon):
        """Bonded floe should be created with particles."""
        floe_id = chrono_sim.create_ice_floe_bonded(large_polygon, thickness=1.0)
        floe = chrono_sim.ice_floes[floe_id]
        
        assert floe.is_bonded
        assert floe.assembly is not None
        assert floe.n_particles > 10
        
    def test_bonded_floe_has_bonds(self, chrono_sim, large_polygon):
        """Bonded floe should have bonds between particles."""
        floe_id = chrono_sim.create_ice_floe_bonded(large_polygon)
        floe = chrono_sim.ice_floes[floe_id]
        
        assert floe.assembly.n_bonds > 0
        
    def test_bonded_floe_initially_one_fragment(self, chrono_sim, large_polygon):
        """Bonded floe should start as one fragment."""
        floe_id = chrono_sim.create_ice_floe_bonded(large_polygon)
        floe = chrono_sim.ice_floes[floe_id]
        
        assert floe.n_fragments == 1


class TestPhysicsStep:
    """Tests for physics simulation stepping."""
    
    def test_step_advances_time(self, chrono_sim):
        """Step should advance simulation time."""
        initial_time = chrono_sim.time
        chrono_sim.step(dt=0.01)
        
        assert chrono_sim.time > initial_time
        assert abs(chrono_sim.time - 0.01) < 0.001
        
    def test_multiple_steps(self, chrono_sim):
        """Multiple steps should accumulate time."""
        for _ in range(100):
            chrono_sim.step(dt=0.01)
            
        assert abs(chrono_sim.time - 1.0) < 0.01
        
    def test_step_with_ship(self, chrono_sim):
        """Simulation should step with ship present."""
        chrono_sim.create_ship((50, 50), 0)
        
        # Should not raise
        for _ in range(100):
            chrono_sim.step(dt=0.01)
            
    def test_step_with_floes(self, chrono_sim, square_vertices):
        """Simulation should step with ice floes."""
        chrono_sim.create_ship((50, 50), 0)
        chrono_sim.create_ice_floe_rigid(square_vertices, thickness=1.0)
        
        # Should not raise
        for _ in range(100):
            chrono_sim.step(dt=0.01)


class TestShipControl:
    """Tests for ship thrust and rudder control."""
    
    def test_apply_thrust_changes_velocity(self, chrono_sim):
        """Applying thrust should accelerate ship."""
        chrono_sim.create_ship((50, 50), 0)  # Heading = 0 means +x direction
        
        initial_state = chrono_sim.get_ship_state()
        
        # Apply thrust and step
        for _ in range(100):
            chrono_sim.apply_ship_thrust(thrust=1e5, rudder=0)
            chrono_sim.step(dt=0.01)
            
        final_state = chrono_sim.get_ship_state()
        
        # Ship should have moved forward (positive x)
        assert final_state['x'] > initial_state['x']
        
    def test_apply_rudder_changes_heading(self, chrono_sim):
        """Applying rudder should rotate ship."""
        chrono_sim.create_ship((50, 50), 0)
        
        initial_state = chrono_sim.get_ship_state()
        
        # Apply rudder torque
        for _ in range(100):
            chrono_sim.apply_ship_thrust(thrust=0, rudder=1e4)
            chrono_sim.step(dt=0.01)
            
        final_state = chrono_sim.get_ship_state()
        
        # Heading should have changed
        assert abs(final_state['heading'] - initial_state['heading']) > 0.01


class TestEnergyAndMomentum:
    """Tests for energy and momentum calculations."""
    
    def test_kinetic_energy_initially_zero(self, chrono_sim):
        """Kinetic energy should be zero at rest."""
        chrono_sim.create_ship((50, 50), 0)
        
        ke = chrono_sim.get_total_kinetic_energy()
        assert ke < 1.0  # Small tolerance for numerical noise
        
    def test_kinetic_energy_increases_with_thrust(self, chrono_sim):
        """Applying thrust should increase kinetic energy."""
        chrono_sim.create_ship((50, 50), 0)
        
        initial_ke = chrono_sim.get_total_kinetic_energy()
        
        for _ in range(100):
            chrono_sim.apply_ship_thrust(thrust=1e5, rudder=0)
            chrono_sim.step(dt=0.01)
            
        final_ke = chrono_sim.get_total_kinetic_energy()
        assert final_ke > initial_ke
        
    def test_momentum_zero_at_rest(self, chrono_sim):
        """Momentum should be zero when at rest."""
        chrono_sim.create_ship((50, 50), 0)
        
        p = chrono_sim.get_total_momentum()
        assert np.linalg.norm(p) < 1.0


class TestFloeState:
    """Tests for getting floe state information."""
    
    def test_get_rigid_floe_state(self, chrono_sim, square_vertices):
        """Should get state for rigid floe."""
        floe_id = chrono_sim.create_ice_floe_rigid(square_vertices)
        
        state = chrono_sim.get_floe_state(floe_id)
        assert state is not None
        assert state['type'] == 'rigid'
        
    def test_get_bonded_floe_state(self, chrono_sim, large_polygon):
        """Should get state for bonded floe."""
        floe_id = chrono_sim.create_ice_floe_bonded(large_polygon)
        
        state = chrono_sim.get_floe_state(floe_id)
        assert state is not None
        assert state['type'] == 'bonded'
        assert 'n_particles' in state
        assert 'n_fragments' in state
        
    def test_get_nonexistent_floe_returns_none(self, chrono_sim):
        """Getting state for invalid ID should return None."""
        state = chrono_sim.get_floe_state(9999)
        assert state is None


class TestFractureEvents:
    """Tests for fracture event tracking."""
    
    def test_initially_no_fracture_events(self, chrono_sim):
        """Should have no fracture events initially."""
        events = chrono_sim.get_fracture_events()
        assert len(events) == 0
        
    def test_get_fracture_events_clears_list(self, chrono_sim):
        """Getting events should clear the internal list."""
        # First call
        events1 = chrono_sim.get_fracture_events()
        # Second call should be empty
        events2 = chrono_sim.get_fracture_events()
        assert len(events2) == 0


class TestCleanup:
    """Tests for resource cleanup."""
    
    def test_cleanup_clears_floes(self, default_config):
        """Cleanup should remove all floes."""
        sim = ChronoIceSimulator(config=default_config)
        sim.create_ice_floe_rigid(np.array([[0,0], [10,0], [10,10], [0,10]]))
        sim.create_ice_floe_rigid(np.array([[20,0], [30,0], [30,10], [20,10]]))
        
        assert len(sim.ice_floes) == 2
        
        sim.cleanup()
        assert len(sim.ice_floes) == 0
        
    def test_cleanup_clears_ship(self, default_config):
        """Cleanup should remove ship."""
        sim = ChronoIceSimulator(config=default_config)
        sim.create_ship((50, 50), 0)
        
        assert sim.ship_body is not None
        
        sim.cleanup()
        assert sim.ship_body is None
