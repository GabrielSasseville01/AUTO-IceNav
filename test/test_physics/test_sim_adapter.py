"""
Unit tests for simulation adapter interface.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import ChronoSimAdapter


class TestAdapterCreation:
    """Tests for adapter initialization."""
    
    def test_create_adapter(self, default_config):
        """Adapter should be created from config."""
        adapter = ChronoSimAdapter(default_config)
        assert adapter is not None
        adapter.cleanup()
        
    def test_adapter_has_simulator(self, chrono_adapter):
        """Adapter should have underlying simulator."""
        assert chrono_adapter.simulator is not None
        
    def test_adapter_uses_config_dt(self, default_config):
        """Adapter should use dt from config."""
        adapter = ChronoSimAdapter(default_config)
        assert adapter.dt == default_config['physics']['dt']
        adapter.cleanup()


class TestInterfaceCompliance:
    """Tests for required interface methods."""
    
    def test_has_load_ice_field(self, chrono_adapter):
        """Adapter should have load_ice_field method."""
        assert hasattr(chrono_adapter, 'load_ice_field')
        assert callable(chrono_adapter.load_ice_field)
        
    def test_has_set_ship_pose(self, chrono_adapter):
        """Adapter should have set_ship_pose method."""
        assert hasattr(chrono_adapter, 'set_ship_pose')
        assert callable(chrono_adapter.set_ship_pose)
        
    def test_has_apply_control(self, chrono_adapter):
        """Adapter should have apply_control method."""
        assert hasattr(chrono_adapter, 'apply_control')
        assert callable(chrono_adapter.apply_control)
        
    def test_has_step(self, chrono_adapter):
        """Adapter should have step method."""
        assert hasattr(chrono_adapter, 'step')
        assert callable(chrono_adapter.step)
        
    def test_has_get_ship_state(self, chrono_adapter):
        """Adapter should have get_ship_state method."""
        assert hasattr(chrono_adapter, 'get_ship_state')
        assert callable(chrono_adapter.get_ship_state)
        
    def test_has_get_observable_ice(self, chrono_adapter):
        """Adapter should have get_observable_ice method."""
        assert hasattr(chrono_adapter, 'get_observable_ice')
        assert callable(chrono_adapter.get_observable_ice)


class TestIceFieldLoading:
    """Tests for loading ice floes."""
    
    def test_load_single_floe(self, chrono_adapter, square_vertices):
        """Should load a single ice floe."""
        chrono_adapter.load_ice_field([square_vertices])
        
        assert chrono_adapter.get_total_floe_count() == 1
        
    def test_load_multiple_floes(self, chrono_adapter, square_vertices):
        """Should load multiple ice floes."""
        polygons = [
            square_vertices,
            square_vertices + [50, 0],
            square_vertices + [100, 0],
        ]
        chrono_adapter.load_ice_field(polygons)
        
        assert chrono_adapter.get_total_floe_count() == 3
        
    def test_load_with_thicknesses(self, chrono_adapter, square_vertices):
        """Should accept custom thicknesses per floe."""
        polygons = [square_vertices, square_vertices + [50, 0]]
        thicknesses = [1.5, 2.0]
        
        # Should not raise
        chrono_adapter.load_ice_field(polygons, thicknesses=thicknesses)
        
    def test_small_floes_are_rigid(self, chrono_adapter, square_vertices):
        """Small floes below threshold should be rigid."""
        # Small 10x10 polygon is below default 500m² threshold
        chrono_adapter.load_ice_field([square_vertices])
        
        # Should have created rigid floe
        floe = list(chrono_adapter.simulator.ice_floes.values())[0]
        assert not floe.is_bonded
        
    def test_large_floes_are_bonded(self, chrono_adapter, large_polygon):
        """Large floes above threshold should be bonded."""
        # 30x30 = 900m² is above default 500m² threshold
        chrono_adapter.load_ice_field([large_polygon])
        
        # Should have created bonded floe
        floe = list(chrono_adapter.simulator.ice_floes.values())[0]
        assert floe.is_bonded


class TestShipPose:
    """Tests for ship position and heading."""
    
    def test_set_ship_pose(self, chrono_adapter):
        """Should set ship position and heading."""
        chrono_adapter.set_ship_pose(100, 50, np.pi/4)
        
        state = chrono_adapter.get_ship_state()
        assert abs(state[0] - 100) < 0.01  # x
        assert abs(state[1] - 50) < 0.01   # y
        assert abs(state[2] - np.pi/4) < 0.01  # psi
        
    def test_get_ship_position(self, chrono_adapter):
        """Should get ship (x, y) position."""
        chrono_adapter.set_ship_pose(100, 50, 0)
        
        pos = chrono_adapter.get_ship_position()
        assert len(pos) == 2
        assert abs(pos[0] - 100) < 0.01
        assert abs(pos[1] - 50) < 0.01


class TestStateVector:
    """Tests for ship state vector format."""
    
    def test_state_vector_shape(self, chrono_adapter):
        """State vector should be length 6."""
        chrono_adapter.set_ship_pose(0, 0, 0)
        
        state = chrono_adapter.get_ship_state()
        assert state.shape == (6,)
        
    def test_state_vector_format(self, chrono_adapter):
        """State should be [x, y, psi, u, v, r]."""
        chrono_adapter.set_ship_pose(100, 50, np.pi/4)
        
        state = chrono_adapter.get_ship_state()
        
        # Position
        assert abs(state[0] - 100) < 0.01  # x
        assert abs(state[1] - 50) < 0.01   # y
        
        # Heading
        assert abs(state[2] - np.pi/4) < 0.01  # psi
        
        # Initial velocities should be ~0
        assert abs(state[3]) < 0.1  # u (surge)
        assert abs(state[4]) < 0.1  # v (sway)
        assert abs(state[5]) < 0.1  # r (yaw rate)


class TestSimulationStep:
    """Tests for stepping the simulation."""
    
    def test_step_advances_time(self, chrono_adapter):
        """Step should advance simulation time."""
        chrono_adapter.set_ship_pose(50, 50, 0)
        
        initial_time = chrono_adapter.get_simulation_time()
        chrono_adapter.step()
        
        assert chrono_adapter.get_simulation_time() > initial_time
        
    def test_control_then_step(self, chrono_adapter):
        """Should apply control before stepping."""
        chrono_adapter.set_ship_pose(50, 50, 0)
        
        # Apply control
        chrono_adapter.apply_control(thrust=1e5, rudder=0)
        
        # Step should not raise
        chrono_adapter.step()


class TestObservableIce:
    """Tests for ice observation interface."""
    
    def test_get_observable_ice_empty(self, chrono_adapter):
        """Should return empty list when no ice."""
        chrono_adapter.set_ship_pose(50, 50, 0)
        
        obs = chrono_adapter.get_observable_ice()
        assert isinstance(obs, list)
        assert len(obs) == 0
        
    def test_get_observable_ice_with_floes(self, chrono_adapter, square_vertices):
        """Should return floe info when ice present."""
        chrono_adapter.load_ice_field([square_vertices])
        chrono_adapter.set_ship_pose(5, 5, 0)  # Inside the square
        
        obs = chrono_adapter.get_observable_ice()
        assert len(obs) == 1
        
        floe_info = obs[0]
        assert 'vertices' in floe_info
        assert 'centroid' in floe_info
        assert 'distance' in floe_info
        
    def test_observable_ice_sorted_by_distance(self, chrono_adapter, square_vertices):
        """Observable ice should be sorted by distance."""
        polygons = [
            square_vertices + [100, 0],  # Far
            square_vertices,              # Close
            square_vertices + [50, 0],   # Medium
        ]
        chrono_adapter.load_ice_field(polygons)
        chrono_adapter.set_ship_pose(5, 5, 0)
        
        obs = chrono_adapter.get_observable_ice()
        
        # Should be sorted by distance
        distances = [o['distance'] for o in obs]
        assert distances == sorted(distances)


class TestFractureEvents:
    """Tests for fracture event retrieval."""
    
    def test_get_fracture_events_empty_initially(self, chrono_adapter):
        """Should return empty list initially."""
        events = chrono_adapter.get_fracture_events()
        assert isinstance(events, list)
        assert len(events) == 0


class TestEnergyState:
    """Tests for energy metrics."""
    
    def test_get_energy_state(self, chrono_adapter):
        """Should return energy state dictionary."""
        chrono_adapter.set_ship_pose(50, 50, 0)
        
        energy = chrono_adapter.get_energy_state()
        
        assert isinstance(energy, dict)
        assert 'kinetic_energy' in energy
        assert 'time' in energy


class TestCleanup:
    """Tests for cleanup."""
    
    def test_cleanup_releases_resources(self, default_config):
        """Cleanup should release resources."""
        adapter = ChronoSimAdapter(default_config)
        adapter.load_ice_field([np.array([[0,0], [10,0], [10,10], [0,10]])])
        
        # Should not raise
        adapter.cleanup()
        
        # After cleanup, floe count should be 0
        assert adapter.get_total_floe_count() == 0
