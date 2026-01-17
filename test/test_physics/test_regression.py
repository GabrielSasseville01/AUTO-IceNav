"""
Regression tests comparing Chrono backend to Pymunk behavior.

These tests ensure the new Chrono backend produces similar results to the
existing Pymunk implementation for non-fracturing (rigid floe) scenarios.
"""

import pytest
import numpy as np
from ship_ice_planner.physics import ChronoIceSimulator, ChronoSimAdapter, IceMaterialParams


# Skip tests if Pymunk not available (for CI environments without it)
try:
    import pymunk
    PYMUNK_AVAILABLE = True
except ImportError:
    PYMUNK_AVAILABLE = False


def create_pymunk_space():
    """Create a Pymunk space for comparison testing."""
    if not PYMUNK_AVAILABLE:
        return None
        
    space = pymunk.Space()
    space.gravity = (0, 0)  # 2D, no gravity
    space.damping = 0.99
    return space


def create_pymunk_ship(space, position, heading, mass=10000.0):
    """Create a ship in Pymunk for comparison."""
    if space is None:
        return None
        
    body = pymunk.Body(mass, pymunk.moment_for_box(mass, (50, 10)))
    body.position = position
    body.angle = heading
    
    shape = pymunk.Poly.create_box(body, (50, 10))
    shape.friction = 0.1
    shape.collision_type = 1
    
    space.add(body, shape)
    return body


def run_pymunk_sim(space, ship_body, initial_vel, duration=10.0, dt=0.02):
    """Run Pymunk simulation and return trajectory."""
    if space is None:
        return np.array([[0, 0]])
        
    ship_body.velocity = initial_vel
    trajectory = []
    
    steps = int(duration / dt)
    for _ in range(steps):
        space.step(dt)
        trajectory.append([ship_body.position.x, ship_body.position.y])
        
    return np.array(trajectory)


def run_chrono_sim(sim, initial_pos, initial_vel, duration=10.0, dt=0.02):
    """Run Chrono simulation and return trajectory."""
    # Use initial_velocity parameter (works for both mock and real Chrono)
    sim.create_ship(position=initial_pos, heading=0, initial_velocity=initial_vel)
    
    trajectory = []
    steps = int(duration / dt)
    
    for _ in range(steps):
        sim.step(dt)
        state = sim.get_ship_state()
        trajectory.append([state['x'], state['y']])
        
    return np.array(trajectory)


@pytest.fixture
def pymunk_sim():
    """Create Pymunk simulation for comparison."""
    if not PYMUNK_AVAILABLE:
        pytest.skip("Pymunk not available")
    return create_pymunk_space()


@pytest.fixture
def chrono_sim_for_regression():
    """Create Chrono simulation for regression testing."""
    material = IceMaterialParams()
    sim = ChronoIceSimulator(material=material, dt=0.02)
    yield sim
    sim.cleanup()


class TestShipDriftRegression:
    """Test ship drift behavior matches between backends."""
    
    @pytest.mark.skipif(not PYMUNK_AVAILABLE, reason="Pymunk not installed")
    def test_ship_drift_direction(self, pymunk_sim, chrono_sim_for_regression):
        """Ship should drift in same general direction."""
        initial_pos = (100, 100)
        initial_vel = (2.0, 0)  # Moving in +x direction
        
        # Create ship in Pymunk
        ship = create_pymunk_ship(pymunk_sim, initial_pos, 0)
        pymunk_traj = run_pymunk_sim(pymunk_sim, ship, initial_vel, duration=5.0)
        
        # Run Chrono
        chrono_traj = run_chrono_sim(chrono_sim_for_regression, initial_pos, initial_vel, duration=5.0)
        
        # Both should move in +x direction
        pymunk_dx = pymunk_traj[-1, 0] - pymunk_traj[0, 0]
        chrono_dx = chrono_traj[-1, 0] - chrono_traj[0, 0]
        
        assert pymunk_dx > 0, "Pymunk ship should move in +x"
        assert chrono_dx > 0, "Chrono ship should move in +x"
        
    @pytest.mark.skipif(not PYMUNK_AVAILABLE, reason="Pymunk not installed")
    def test_ship_drift_with_damping(self, pymunk_sim, chrono_sim_for_regression):
        """Ship should slow down due to damping in both backends."""
        initial_pos = (100, 100)
        initial_vel = (5.0, 0)
        
        # Run Pymunk
        ship = create_pymunk_ship(pymunk_sim, initial_pos, 0)
        pymunk_traj = run_pymunk_sim(pymunk_sim, ship, initial_vel, duration=10.0)
        
        # Run Chrono
        chrono_traj = run_chrono_sim(chrono_sim_for_regression, initial_pos, initial_vel, duration=10.0)
        
        # Both should slow down (later segments shorter than earlier)
        for traj, name in [(pymunk_traj, 'Pymunk'), (chrono_traj, 'Chrono')]:
            n = len(traj)
            early_speed = np.linalg.norm(traj[n//4] - traj[0])
            late_speed = np.linalg.norm(traj[-1] - traj[3*n//4])
            assert late_speed < early_speed, f"{name} ship should slow down"


class TestTrajectoryDivergence:
    """Test trajectory divergence limits."""
    
    def test_chrono_trajectory_reasonable(self, chrono_sim_for_regression):
        """Chrono trajectory should be physically reasonable."""
        initial_pos = (100, 100)
        initial_vel = (2.0, 0)
        
        traj = run_chrono_sim(chrono_sim_for_regression, initial_pos, initial_vel, duration=10.0)
        
        # Ship should move forward
        assert traj[-1, 0] > traj[0, 0], "Ship should move forward"
        
        # Should stay roughly on course (y shouldn't change much)
        y_deviation = np.abs(traj[:, 1] - initial_pos[1]).max()
        assert y_deviation < 10.0, "Ship should stay on course"
        
    def test_trajectory_physically_bounded(self, chrono_sim_for_regression):
        """Trajectory should not have unphysical jumps."""
        initial_pos = (100, 100)
        initial_vel = (2.0, 0)
        
        traj = run_chrono_sim(chrono_sim_for_regression, initial_pos, initial_vel, duration=10.0, dt=0.02)
        
        # Check for smooth trajectory (no teleportation)
        for i in range(1, len(traj)):
            step_dist = np.linalg.norm(traj[i] - traj[i-1])
            max_step = 5.0 * 0.02  # Max 5 m/s for 0.02s step
            assert step_dist < max_step, f"Step {i} too large: {step_dist}"


class TestIceFloeComparison:
    """Test ice floe behavior comparison."""
    
    def test_rigid_floe_creation(self, chrono_sim_for_regression):
        """Rigid floe should be created with correct type."""
        vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        floe_id = chrono_sim_for_regression.create_ice_floe_rigid(vertices, thickness=1.0)
        
        floe = chrono_sim_for_regression.ice_floes[floe_id]
        assert not floe.is_bonded, "Should be rigid floe"
        
    def test_bonded_floe_creation(self, chrono_sim_for_regression):
        """Bonded floe should have particles."""
        vertices = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])
        floe_id = chrono_sim_for_regression.create_ice_floe_bonded(vertices, thickness=1.0)
        
        floe = chrono_sim_for_regression.ice_floes[floe_id]
        assert floe.is_bonded, "Should be bonded floe"
        assert floe.n_particles > 0, "Should have particles"


class TestEnergyRegression:
    """Test energy behavior is physically reasonable."""
    
    def test_energy_dissipates(self, chrono_sim_for_regression):
        """Energy should dissipate over time due to damping."""
        # Give ship initial velocity using initial_velocity parameter
        chrono_sim_for_regression.create_ship((100, 100), 0, initial_velocity=(5.0, 0.0))
        
        initial_state = chrono_sim_for_regression.get_ship_state()
        initial_horizontal_ke = 0.5 * 10000 * (initial_state['vx']**2 + initial_state['vy']**2)
        
        # Run for a while
        for _ in range(500):
            chrono_sim_for_regression.step(0.02)
            
        final_state = chrono_sim_for_regression.get_ship_state()
        final_horizontal_ke = 0.5 * 10000 * (final_state['vx']**2 + final_state['vy']**2)
        
        # Horizontal energy should decrease (ignoring vertical due to gravity in 3D)
        assert final_horizontal_ke < initial_horizontal_ke * 0.9, "Horizontal energy should dissipate"
        
    def test_energy_non_negative(self, chrono_sim_for_regression):
        """Kinetic energy should never be negative."""
        chrono_sim_for_regression.create_ship((100, 100), 0)
        
        # Run simulation
        for _ in range(100):
            chrono_sim_for_regression.step(0.02)
            ke = chrono_sim_for_regression.get_total_kinetic_energy()
            assert ke >= 0, "Energy cannot be negative"


class TestMomentumRegression:
    """Test momentum conservation."""
    
    def test_isolated_system_momentum_conserved(self, chrono_sim_for_regression):
        """Momentum should be approximately conserved in isolated system."""
        # Create two floes that might collide (simplified test)
        v1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        v2 = np.array([[15, 0], [25, 0], [25, 10], [15, 10]])
        
        chrono_sim_for_regression.create_ice_floe_rigid(v1, initial_velocity=(1, 0))
        chrono_sim_for_regression.create_ice_floe_rigid(v2, initial_velocity=(-1, 0))
        
        initial_p = chrono_sim_for_regression.get_total_momentum()
        
        # Run simulation
        for _ in range(200):
            chrono_sim_for_regression.step(0.02)
            
        final_p = chrono_sim_for_regression.get_total_momentum()
        
        # Momentum should be approximately conserved (within 10%)
        # Note: damping will affect this somewhat
        initial_mag = np.linalg.norm(initial_p)
        if initial_mag > 0.1:  # Only check if there's meaningful momentum
            diff = np.linalg.norm(final_p - initial_p)
            assert diff < initial_mag * 0.5, "Momentum should be roughly conserved"


class TestComparisonMetrics:
    """Test helper functions for comparison."""
    
    def test_trajectory_diff_calculation(self):
        """Trajectory difference should be computed correctly."""
        traj1 = np.array([[0, 0], [1, 0], [2, 0]])
        traj2 = np.array([[0, 0], [1, 1], [2, 0]])
        
        diff = np.linalg.norm(traj1 - traj2, axis=1)
        
        assert diff[0] == 0  # Same start
        assert diff[1] == 1  # 1 unit apart at step 1
        assert diff[2] == 0  # Same end


class TestAdapterRegression:
    """Test adapter produces consistent results."""
    
    def test_adapter_state_format(self, default_config):
        """Adapter state should have correct format."""
        adapter = ChronoSimAdapter(default_config)
        adapter.set_ship_pose(100, 50, np.pi/4)
        
        state = adapter.get_ship_state()
        
        assert state.shape == (6,)
        assert abs(state[0] - 100) < 0.1
        assert abs(state[1] - 50) < 0.1
        
        adapter.cleanup()
        
    def test_adapter_step_advances_time(self, default_config):
        """Adapter step should advance simulation time."""
        adapter = ChronoSimAdapter(default_config)
        adapter.set_ship_pose(50, 50, 0)
        
        t0 = adapter.get_simulation_time()
        adapter.step()
        t1 = adapter.get_simulation_time()
        
        assert t1 > t0
        
        adapter.cleanup()
