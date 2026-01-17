"""
Integration tests for full simulation pipeline with Chrono backend.

These tests verify that the Chrono physics backend integrates properly
with the rest of the simulation infrastructure.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import ChronoSimAdapter, IceMaterialParams


@pytest.fixture
def simulation_config():
    """Configuration for integration tests."""
    return {
        'physics': {
            'backend': 'chrono',
            'use_gpu': False,
            'dt': 0.02,
        },
        'ice_material': {
            'density': 920.0,
            'youngs_modulus': 5.0e9,
            'tensile_strength': 0.5e6,
            'shear_strength': 0.4e6,
            'thickness': 1.0,
        },
        'bonded_dem': {
            'particle_size_ratio': 0.08,
            'min_floe_area_for_bonding': 500.0,
        },
        'ship': {
            'mass': 10000.0,
            'length': 50.0,
            'beam': 10.0,
        },
    }


@pytest.fixture
def ice_field_polygons():
    """Sample ice field polygons for testing."""
    return [
        # Small rigid floes
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        np.array([[20, 0], [30, 0], [30, 10], [20, 10]]),
        np.array([[40, 0], [50, 0], [50, 10], [40, 10]]),
        # Large bonded floe
        np.array([[100, 100], [140, 100], [140, 140], [100, 140]]),
    ]


class TestFullSimulationPipeline:
    """Tests for complete simulation runs."""
    
    def test_simulation_runs_to_completion(self, simulation_config, ice_field_polygons):
        """Full simulation should complete without errors."""
        adapter = ChronoSimAdapter(simulation_config)
        
        # Load ice field
        adapter.load_ice_field(ice_field_polygons)
        
        # Set ship pose
        adapter.set_ship_pose(50, 50, 0)
        
        # Run for simulated 10 seconds
        steps = 500  # 500 * 0.02 = 10 seconds
        for _ in range(steps):
            adapter.apply_control(thrust=1e4, rudder=0)
            adapter.step()
            
        # Should complete without error
        final_state = adapter.get_ship_state()
        assert final_state is not None
        
        adapter.cleanup()
        
    def test_simulation_time_advances(self, simulation_config):
        """Simulation time should advance correctly."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(50, 50, 0)
        
        initial_time = adapter.get_simulation_time()
        
        for _ in range(100):
            adapter.step()
            
        final_time = adapter.get_simulation_time()
        
        expected_time = 100 * simulation_config['physics']['dt']
        assert abs(final_time - initial_time - expected_time) < 0.01
        
        adapter.cleanup()


class TestPlannerIntegration:
    """Tests for planner state format compatibility."""
    
    def test_ship_state_format_for_planner(self, simulation_config):
        """Ship state should have format expected by planners."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(100, 50, np.pi/4)
        
        state = adapter.get_ship_state()
        
        # Planners expect [x, y, psi, u, v, r]
        assert len(state) == 6
        
        # Position
        assert isinstance(state[0], (int, float))  # x
        assert isinstance(state[1], (int, float))  # y
        
        # Heading
        assert isinstance(state[2], (int, float))  # psi
        assert -np.pi <= state[2] <= 2*np.pi  # Valid angle range
        
        # Velocities
        assert isinstance(state[3], (int, float))  # u
        assert isinstance(state[4], (int, float))  # v
        assert isinstance(state[5], (int, float))  # r
        
        adapter.cleanup()
        
    def test_observable_ice_format(self, simulation_config, ice_field_polygons):
        """Observable ice should have format expected by planners."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.load_ice_field(ice_field_polygons)
        adapter.set_ship_pose(5, 5, 0)  # Near the floes
        
        obstacles = adapter.get_observable_ice()
        
        assert isinstance(obstacles, list)
        
        if len(obstacles) > 0:
            obs = obstacles[0]
            
            # Required fields for planners
            assert 'vertices' in obs
            assert 'centroid' in obs
            assert isinstance(obs['vertices'], np.ndarray)
            assert obs['vertices'].ndim == 2
            assert obs['vertices'].shape[1] == 2
            
        adapter.cleanup()


class TestControllerIntegration:
    """Tests for controller command application."""
    
    def test_thrust_control_moves_ship(self, simulation_config):
        """Applying thrust should move ship forward."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(100, 100, 0)  # Heading = 0 means +x
        
        initial_state = adapter.get_ship_state()
        
        # Apply forward thrust
        for _ in range(200):
            adapter.apply_control(thrust=1e5, rudder=0)
            adapter.step()
            
        final_state = adapter.get_ship_state()
        
        # Ship should have moved in +x direction
        assert final_state[0] > initial_state[0]
        
        adapter.cleanup()
        
    def test_rudder_control_turns_ship(self, simulation_config):
        """Applying rudder should change ship heading."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(100, 100, 0)
        
        initial_heading = adapter.get_ship_state()[2]
        
        # Apply rudder torque
        for _ in range(200):
            adapter.apply_control(thrust=0, rudder=1e4)
            adapter.step()
            
        final_heading = adapter.get_ship_state()[2]
        
        # Heading should have changed
        assert abs(final_heading - initial_heading) > 0.01
        
        adapter.cleanup()


class TestCollisionHandling:
    """Tests for ship-ice collision handling."""
    
    def test_ship_ice_collision_scenario(self, simulation_config):
        """Ship should interact with ice floes."""
        adapter = ChronoSimAdapter(simulation_config)
        
        # Place ice floe directly in front of ship
        ice_polygon = np.array([[150, 95], [170, 95], [170, 105], [150, 105]])
        adapter.load_ice_field([ice_polygon])
        
        # Ship starts at (100, 100) heading toward floe
        adapter.set_ship_pose(100, 100, 0)
        
        # Run toward floe
        for _ in range(300):
            adapter.apply_control(thrust=1e5, rudder=0)
            adapter.step()
            
        # Should complete without error
        final_state = adapter.get_ship_state()
        assert final_state is not None
        
        adapter.cleanup()


class TestFractureIntegration:
    """Tests for fracture event handling."""
    
    def test_fracture_events_retrievable(self, simulation_config):
        """Fracture events should be retrievable from adapter."""
        adapter = ChronoSimAdapter(simulation_config)
        
        # Load bonded floe
        large_floe = np.array([[0, 0], [35, 0], [35, 35], [0, 35]])
        adapter.load_ice_field([large_floe])
        adapter.set_ship_pose(50, 17, np.pi)  # Ship heading toward floe
        
        # Run simulation
        for _ in range(100):
            adapter.apply_control(thrust=5e5, rudder=0)
            adapter.step()
            
        events = adapter.get_fracture_events()
        assert isinstance(events, list)
        
        adapter.cleanup()
        
    def test_fragment_count_tracking(self, simulation_config):
        """Fragment count should be tracked correctly."""
        adapter = ChronoSimAdapter(simulation_config)
        
        # Load bonded floe
        large_floe = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])
        adapter.load_ice_field([large_floe])
        
        initial_fragments = adapter.get_fragment_count()
        
        # Fragment count should be at least 1
        assert initial_fragments >= 1
        
        adapter.cleanup()


class TestDataRecording:
    """Tests for simulation data recording."""
    
    def test_collision_data_recorded(self, simulation_config):
        """Collision data should be recorded."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(50, 50, 0)
        
        # Run simulation
        for _ in range(100):
            adapter.step()
            
        impulses = adapter.get_collision_impulses()
        assert isinstance(impulses, list)
        
        adapter.cleanup()
        
    def test_energy_state_recorded(self, simulation_config):
        """Energy state should be available."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(50, 50, 0)
        
        energy = adapter.get_energy_state()
        
        assert isinstance(energy, dict)
        assert 'kinetic_energy' in energy
        assert energy['kinetic_energy'] >= 0
        
        adapter.cleanup()


class TestPhysicsValidation:
    """Tests for physical correctness."""
    
    def test_ship_stays_bounded(self, simulation_config):
        """Ship should not teleport or explode (XY plane smoothness check)."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(0, 0, 0)
        
        positions = []
        
        # Run fewer steps with gentle thrust to limit gravity effects in 3D
        for _ in range(50):
            adapter.apply_control(thrust=1e4, rudder=0)  # Gentle forward thrust only
            adapter.step()
            
            state = adapter.get_ship_state()
            positions.append([state[0], state[1]])
            
        positions = np.array(positions)
        
        # Check XY trajectory is smooth (no instantaneous teleportation)
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            # XY movement at reasonable thrust should be bounded
            max_dist = 1.0  # 1m per step is very generous for 0.02s step
            assert dist < max_dist, f"Ship teleported at step {i}: moved {dist:.4f}m"
            
        adapter.cleanup()
        
    def test_velocities_bounded(self, simulation_config):
        """Velocities should remain physically reasonable for XY motion."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.set_ship_pose(0, 0, 0)
        
        # Test XY velocities with gentle thrust (not affected by gravity)
        max_xy_speed = 25.0  # m/s - reasonable for gentle propulsion
        
        # Run short simulation with gentle thrust
        for _ in range(50):
            adapter.apply_control(thrust=1e4, rudder=0)  # Gentle thrust
            adapter.step()
            
            state = adapter.get_ship_state()
            xy_speed = np.sqrt(state[3]**2 + state[4]**2)
            assert xy_speed < max_xy_speed, f"Unrealistic XY speed: {xy_speed}"
            
        adapter.cleanup()


class TestLongRunningSimulation:
    """Tests for extended simulation runs."""
    
    @pytest.mark.slow
    def test_extended_simulation_stable(self, simulation_config, ice_field_polygons):
        """Extended simulation should remain stable."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.load_ice_field(ice_field_polygons)
        adapter.set_ship_pose(50, 50, 0)
        
        # Run for 60 simulated seconds
        steps = 3000  # 3000 * 0.02 = 60 seconds
        
        for i in range(steps):
            # Varying controls
            thrust = 1e4 * np.sin(i * 0.01)
            rudder = 1e3 * np.cos(i * 0.02)
            
            adapter.apply_control(thrust=thrust, rudder=rudder)
            adapter.step()
            
            # Periodically check state
            if i % 500 == 0:
                state = adapter.get_ship_state()
                assert not np.any(np.isnan(state)), f"NaN in state at step {i}"
                assert not np.any(np.isinf(state)), f"Inf in state at step {i}"
                
        adapter.cleanup()


class TestMemoryManagement:
    """Tests for memory and resource management."""
    
    def test_cleanup_releases_resources(self, simulation_config):
        """Cleanup should release all resources."""
        adapter = ChronoSimAdapter(simulation_config)
        adapter.load_ice_field([np.array([[0,0], [10,0], [10,10], [0,10]])])
        adapter.set_ship_pose(50, 50, 0)
        
        # Run some steps
        for _ in range(100):
            adapter.step()
            
        # Cleanup
        adapter.cleanup()
        
        # Verify cleanup
        assert adapter.get_total_floe_count() == 0
        
    def test_multiple_adapter_instances(self, simulation_config):
        """Multiple adapter instances should work independently."""
        adapter1 = ChronoSimAdapter(simulation_config)
        adapter2 = ChronoSimAdapter(simulation_config)
        
        adapter1.set_ship_pose(0, 0, 0)
        adapter2.set_ship_pose(100, 100, np.pi)
        
        # Step both
        for _ in range(50):
            adapter1.apply_control(thrust=1e4, rudder=0)
            adapter2.apply_control(thrust=-1e4, rudder=1e3)
            adapter1.step()
            adapter2.step()
            
        state1 = adapter1.get_ship_state()
        state2 = adapter2.get_ship_state()
        
        # States should be different
        assert state1[0] != state2[0] or state1[1] != state2[1]
        
        adapter1.cleanup()
        adapter2.cleanup()
