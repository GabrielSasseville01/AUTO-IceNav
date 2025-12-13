"""Generate training dataset for neural MPC planner
This script generates random discrete A* plans and their corresponding continuous paths
using ship dynamics simulation to create a supervised learning dataset.
Now learns spline parameters instead of full trajectories.
"""
import pickle
import numpy as np
from tqdm import tqdm
import random
from scipy.interpolate import splrep, splev, BSpline

from ship_ice_planner import *
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.utils.sim_utils import generate_obstacles
from ship_ice_planner.utils.utils import DotDict, compute_path_length

# Configuration
CFG = DotDict.load_from_file(FULL_SCALE_SIM_PARAM_CONFIG)
CFG.output_dir = None

# Dataset generation parameters
NUM_SAMPLES = 2000  # Number of training samples to generate (increased for rich data)
MAP_SHAPE = CFG.map_shape  # (height, width) in meters

# Long horizon parameters - support paths from short to very long
MIN_GOAL_DISTANCE = 150.0   # Minimum distance to goal (short horizon)
MAX_GOAL_DISTANCE = 2500.0  # Maximum distance to goal (long horizon - increased)
HORIZON_CATEGORIES = [
    (150.0, 400.0, 'short'),    # Short horizon: 150-400m
    (400.0, 800.0, 'medium'),   # Medium horizon: 400-800m
    (800.0, 1500.0, 'long'),    # Long horizon: 800-1500m
    (1500.0, 2500.0, 'very_long')  # Very long horizon: 1500-2500m
]

# Ice density parameters - ensure high density coverage
MIN_ICE_CONCENTRATION = 0.3  # Minimum ice concentration (30%)
MAX_ICE_CONCENTRATION = 0.8  # Maximum ice concentration (80%)
TARGET_CONCENTRATION_TOLERANCE = 0.05  # Tolerance for concentration

# Obstacle generation parameters - increased for high density
OBSTACLE_MIN_R = 3   # Smaller min radius for dense packing
OBSTACLE_MAX_R = 60  # Larger max radius for variety
MIN_OBSTACLES_FACTOR = 0.15  # Minimum obstacles per 100m^2
MAX_OBSTACLES_FACTOR = 0.35  # Maximum obstacles per 100m^2

# Ship dynamics parameters (for simulation)
DT = CFG.sim_dynamics.dt
TARGET_SPEED = CFG.sim_dynamics.target_speed

# Simulation parameters for long horizons
MAX_SIM_STEPS_SHORT = 3000
MAX_SIM_STEPS_MEDIUM = 6000
MAX_SIM_STEPS_LONG = 12000
MAX_SIM_STEPS_VERY_LONG = 20000


def compute_ice_concentration(obstacles_dict, min_y, max_y, map_width):
    """Compute ice concentration from obstacles"""
    if not obstacles_dict:
        return 0.0
    
    total_area = 0.0
    for ob in obstacles_dict:
        if 'area' in ob:
            total_area += ob['area']
        elif 'vertices' in ob:
            from ship_ice_planner.geometry.polygon import poly_area
            total_area += poly_area(ob['vertices'])
    
    ice_region_area = (max_y - min_y) * map_width
    if ice_region_area > 0:
        return total_area / ice_region_area
    return 0.0


def generate_obstacles_with_concentration(cfg, start_y, goal_y, target_concentration, seed=None, max_attempts=10):
    """Generate obstacles to achieve target ice concentration"""
    from ship_ice_planner.geometry.polygon import poly_area
    
    # Ensure minimum ice region size
    # For short horizons, use smaller margins
    distance = goal_y - start_y
    if distance < 200:
        margin = 10  # Very small margin for short distances
    elif distance < 500:
        margin = 15
    else:
        margin = 20
    
    min_y = max(start_y + margin, 0)
    max_y = min(goal_y - margin, MAP_SHAPE[0])
    
    # Ensure minimum region size
    min_region_height = 40.0  # Minimum 40m vertical space
    if max_y - min_y < min_region_height:
        # Expand region symmetrically
        center = (start_y + goal_y) / 2
        half_height = min_region_height / 2
        min_y = max(0, center - half_height)
        max_y = min(MAP_SHAPE[0], center + half_height)
        
        # If still too small, use a larger default region
        if max_y - min_y < min_region_height:
            min_y = max(0, MAP_SHAPE[0] * 0.1)
            max_y = min(MAP_SHAPE[0], MAP_SHAPE[0] * 0.9)
    
    ice_region_area = (max_y - min_y) * MAP_SHAPE[1]
    
    if ice_region_area < 1000.0:  # Need at least 1000 m^2
        # Fallback: use simpler generation for very small regions
        num_obs = max(5, min(30, int(ice_region_area * target_concentration / 400.0)))
        
        # Scale radius based on region size
        region_height = max_y - min_y
        region_width = MAP_SHAPE[1]
        max_r_small = min(OBSTACLE_MAX_R, int(region_height / 3), int(region_width / 6))
        min_r_small = max(2, OBSTACLE_MIN_R)  # Minimum radius
        
        if max_r_small < min_r_small:
            max_r_small = min_r_small + 2
        
        try:
            obs_dict, obs_verts = generate_obstacles(
                num_obs=num_obs,
                min_r=min_r_small,
                max_r=max_r_small,
                min_x=0,
                max_x=MAP_SHAPE[1],
                min_y=min_y,
                max_y=max_y,
                allow_overlap=False,
                seed=seed
            )
            if obs_dict and len(obs_dict) > 0:
                return obs_dict, obs_verts
        except Exception as e:
            # If that fails, try with even smaller obstacles and allow overlap
            try:
                obs_dict, obs_verts = generate_obstacles(
                    num_obs=min(num_obs, 15),
                    min_r=2,
                    max_r=min(max_r_small, 15),
                    min_x=0,
                    max_x=MAP_SHAPE[1],
                    min_y=min_y,
                    max_y=max_y,
                    allow_overlap=True,  # Allow overlap for very small regions
                    seed=seed
                )
                if obs_dict and len(obs_dict) > 0:
                    return obs_dict, obs_verts
            except Exception:
                pass
    
    # Calculate target number of obstacles based on concentration
    # Estimate: avg obstacle area ~ 400 m^2 (typical floe size)
    target_ice_area = ice_region_area * target_concentration
    avg_obstacle_area = 400.0  # Average area per obstacle
    initial_num_obs = max(10, int(target_ice_area / avg_obstacle_area))
    initial_num_obs = min(initial_num_obs, 200)  # Limit max obstacles
    
    obstacles_dict = []
    obstacles_vertices = []
    
    for attempt in range(max_attempts):
        # Generate obstacles
        num_obs = int(initial_num_obs * (1.0 + 0.2 * (attempt)))  # Increase on each attempt
        num_obs = min(num_obs, 200)  # Cap at reasonable maximum
        
        # Adjust radius limits based on region size
        region_height = max_y - min_y
        region_width = MAP_SHAPE[1]
        max_r_scaled = min(OBSTACLE_MAX_R, int(region_height / 4), int(region_width / 8))
        min_r_scaled = max(OBSTACLE_MIN_R, 2)
        
        if max_r_scaled < min_r_scaled:
            max_r_scaled = min_r_scaled + 5
        
        try:
            obs_dict, obs_verts = generate_obstacles(
                num_obs=num_obs,
                min_r=min_r_scaled,
                max_r=max_r_scaled,
                min_x=0,
                max_x=MAP_SHAPE[1],
                min_y=min_y,
                max_y=max_y,
                allow_overlap=False,
                seed=seed + attempt if seed is not None else None
            )
        except Exception as e:
            # If generation fails, try with smaller obstacles or fewer obstacles
            if attempt < max_attempts - 1:
                num_obs = int(num_obs * 0.7)  # Reduce number
                max_r_scaled = max(min_r_scaled + 1, int(max_r_scaled * 0.8))  # Reduce max radius
                min_r_scaled = max(2, int(min_r_scaled * 0.9))  # Also reduce min radius slightly
                continue
            else:
                # Last attempt failed, will try fallback
                obs_dict = []
                obs_verts = []
                break
        
        if not obs_dict:
            continue
        
        # Filter obstacles to ensure they're in the ice region
        filtered_dict = []
        filtered_verts = []
        for i, ob in enumerate(obs_dict):
            if 'vertices' in ob:
                verts = ob['vertices']
            elif i < len(obs_verts):
                verts = obs_verts[i]
            elif 'centre' in ob:
                # Create a simple circular obstacle
                from ship_ice_planner.geometry.polygon import generate_polygon
                r = ob.get('radius', OBSTACLE_MAX_R / 2)
                center = ob['centre']
                n_verts = 8
                verts = generate_polygon(center[0], center[1], r, n_verts)
            else:
                continue
            
            if verts is None or len(verts) == 0:
                continue
            
            # Check if obstacle is in ice region
            if np.any((verts[:, 1] >= min_y) & (verts[:, 1] <= max_y)):
                filtered_dict.append(ob)
                filtered_verts.append(verts)
        
        if not filtered_dict:
            continue
        
        # Calculate actual concentration
        total_area = sum(poly_area(ob['vertices']) if 'vertices' in ob and ob['vertices'] is not None else 
                        poly_area(filtered_verts[j]) for j, ob in enumerate(filtered_dict))
        actual_concentration = total_area / ice_region_area if ice_region_area > 0 else 0.0
        
        # Check if concentration is acceptable
        if abs(actual_concentration - target_concentration) <= TARGET_CONCENTRATION_TOLERANCE:
            obstacles_dict = filtered_dict
            obstacles_vertices = filtered_verts
            break
        elif actual_concentration < target_concentration - TARGET_CONCENTRATION_TOLERANCE:
            # Need more obstacles
            initial_num_obs = int(num_obs * (target_concentration / max(actual_concentration, 0.01)))
            continue
        else:
            # Too many obstacles, but accept if close enough
            if actual_concentration <= target_concentration + 0.1:
                obstacles_dict = filtered_dict
                obstacles_vertices = filtered_verts
                break
    
    # If we didn't get enough, use what we have or try simple fallback
    if not obstacles_dict:
        # Fallback: generate with minimum number and allow overlap if needed
        num_obs = max(10, min(50, int(ice_region_area * target_concentration / avg_obstacle_area)))
        
        # Scale radius for small regions
        region_height = max_y - min_y
        max_r_fallback = min(OBSTACLE_MAX_R, int(region_height / 3))
        min_r_fallback = max(OBSTACLE_MIN_R, 2)
        
        if max_r_fallback < min_r_fallback:
            max_r_fallback = min_r_fallback + 5
        
        try:
            # First try without overlap
            obs_dict, obs_verts = generate_obstacles(
                num_obs=num_obs,
                min_r=min_r_fallback,
                max_r=max_r_fallback,
                min_x=0,
                max_x=MAP_SHAPE[1],
                min_y=min_y,
                max_y=max_y,
                allow_overlap=False,
                seed=seed
            )
        except Exception:
            # If that fails, try with overlap allowed
            try:
                obs_dict, obs_verts = generate_obstacles(
                    num_obs=min(num_obs, 30),  # Fewer if allowing overlap
                    min_r=min_r_fallback,
                    max_r=max_r_fallback,
                    min_x=0,
                    max_x=MAP_SHAPE[1],
                    min_y=min_y,
                    max_y=max_y,
                    allow_overlap=True,  # Allow overlap as last resort
                    seed=seed
                )
            except Exception as e:
                # Last resort: return empty, caller will handle
                print(f"Warning: Could not generate obstacles for region {min_y:.1f}-{max_y:.1f}m: {e}")
                obs_dict = []
                obs_verts = []
        
        if obs_dict:
            obstacles_dict = obs_dict
            obstacles_vertices = obs_verts
    
    return obstacles_dict, obstacles_vertices


def generate_random_scenario(cfg, seed=None, horizon_category=None):
    """Generate a random scenario with obstacles and start/goal"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Select horizon category if not specified
    if horizon_category is None:
        category = np.random.choice(len(HORIZON_CATEGORIES))
        min_dist, max_dist, category_name = HORIZON_CATEGORIES[category]
    else:
        min_dist, max_dist, category_name = horizon_category
    
    # Random start position (x, y, psi)
    start_x = np.random.uniform(MAP_SHAPE[1] * 0.15, MAP_SHAPE[1] * 0.85)
    start_y = np.random.uniform(0, MAP_SHAPE[0] * 0.08)
    start_psi = np.random.uniform(0, 2 * np.pi)
    start_pose = (start_x, start_y, start_psi)
    
    # Random goal distance based on horizon category
    goal_y = np.random.uniform(start_y + min_dist, 
                               min(start_y + max_dist, MAP_SHAPE[0] * 0.92))
    goal_x = np.random.uniform(MAP_SHAPE[1] * 0.15, MAP_SHAPE[1] * 0.85)
    goal = [goal_x, goal_y]
    
    # Random ice concentration (high density range)
    target_concentration = np.random.uniform(MIN_ICE_CONCENTRATION, MAX_ICE_CONCENTRATION)
    
    # Generate obstacles with target concentration
    obstacles_dict, obstacles_vertices = generate_obstacles_with_concentration(
        cfg, start_y, goal_y, target_concentration, seed=seed
    )
    
    # Calculate masses from areas
    from ship_ice_planner.geometry.polygon import poly_area
    masses = []
    for i, ob in enumerate(obstacles_dict):
        if 'mass' in ob:
            masses.append(ob['mass'])
        elif 'area' in ob:
            masses.append(ob['area'] * 0.1)  # kg per m^2
        else:
            # Calculate from vertices
            if i < len(obstacles_vertices):
                area = poly_area(obstacles_vertices[i])
                masses.append(area * 0.1)  # kg per m^2
            else:
                masses.append(100.0)  # Default mass
    
    # Verify we have obstacles
    if not obstacles_vertices:
        # Fallback: generate minimum obstacles
        num_obs = max(15, int((goal_y - start_y) * MAP_SHAPE[1] * MIN_OBSTACLES_FACTOR / 100))
        obs_dict, obs_verts = generate_obstacles(
            num_obs=num_obs,
            min_r=OBSTACLE_MIN_R,
            max_r=OBSTACLE_MAX_R,
            min_x=0,
            max_x=MAP_SHAPE[1],
            min_y=start_y + 30,
            max_y=goal_y - 30,
            allow_overlap=False,
            seed=seed
        )
        if obs_dict:
            obstacles_dict = obs_dict
            obstacles_vertices = obs_verts
            masses = [ob.get('mass', ob.get('area', 100) * 0.1) if isinstance(ob, dict) else 100.0 
                     for ob in obstacles_dict]
    
    return start_pose, goal, obstacles_vertices, masses, obstacles_dict, category_name, target_concentration


def fit_spline_to_path(path, num_control_points=10, degree=3):
    """
    Fit a B-spline to a path and return control points.
    
    Args:
        path: (N, 3) path array [x, y, psi]
        num_control_points: Number of control points for the spline
        degree: Degree of the B-spline (3 = cubic)
    
    Returns:
        control_points: (num_control_points, 2) control points for x, y
        knots: knot vector
        t: parameter values
    """
    if len(path) < 4:
        # Too few points, return simple linear interpolation
        control_points = np.zeros((num_control_points, 2))
        if len(path) == 1:
            control_points[:, 0] = path[0, 0]
            control_points[:, 1] = path[0, 1]
        else:
            for i in range(num_control_points):
                alpha = i / (num_control_points - 1) if num_control_points > 1 else 0
                idx = min(int(alpha * (len(path) - 1)), len(path) - 1)
                control_points[i] = path[idx, :2]
        # Create simple knot vector
        knots = np.linspace(0, 1, num_control_points + degree + 1)
        t = np.linspace(0, 1, len(path))
        return control_points, knots, t
    
    # Compute arc length parameterization
    positions = path[:, :2]
    distances = compute_path_length(positions, cumsum=True)
    total_dist = distances[-1]
    
    if total_dist < 1e-6:
        # Path is essentially a point
        control_points = np.zeros((num_control_points, 2))
        control_points[:, 0] = path[0, 0]
        control_points[:, 1] = path[0, 1]
        knots = np.linspace(0, 1, num_control_points + degree + 1)
        t = np.linspace(0, 1, len(path))
        return control_points, knots, t
    
    # Normalize distances to [0, 1]
    t_normalized = distances / total_dist
    
    # Fit spline for x and y separately
    try:
        # Use splrep to fit B-spline
        tck_x, u_x = splrep(t_normalized, path[:, 0], k=min(degree, len(path)-1), s=0)
        tck_y, u_y = splrep(t_normalized, path[:, 1], k=min(degree, len(path)-1), s=0)
        
        # Extract control points (coefficients)
        knots_x, coeffs_x, degree_x = tck_x
        knots_y, coeffs_y, degree_y = tck_y
        
        # For uniform representation, use the same number of control points
        # Evaluate spline at uniform parameter values to get control points
        u_uniform = np.linspace(0, 1, num_control_points)
        control_x = splev(u_uniform, tck_x)
        control_y = splev(u_uniform, tck_y)
        control_points = np.column_stack([control_x, control_y])
        
        # Use knots from x (they should be similar)
        knots = knots_x
        
        return control_points, knots, t_normalized
    except Exception as e:
        # Fallback: use evenly spaced points from path
        indices = np.linspace(0, len(path) - 1, num_control_points, dtype=int)
        control_points = path[indices, :2]
        knots = np.linspace(0, 1, num_control_points + degree + 1)
        t = np.linspace(0, 1, len(path))
        return control_points, knots, t


def get_discrete_and_continuous_plan(cfg, obstacles, masses, start_pose, goal_y, num_spline_control_points=15):
    """Get discrete A* plan and continuous path, then extract spline parameters"""
    ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
    prim = Primitives(**cfg.prim)
    swath_dict, swath_shape = generate_swath(ship, prim)
    
    costmap = CostMap(length=cfg.map_shape[0],
                     width=cfg.map_shape[1],
                     ship_mass=ship.mass,
                     padding=swath_shape[0] // 2,
                     **cfg.costmap)
    
    # Update costmap
    start_scaled = (start_pose[0] * cfg.costmap.scale, 
                    start_pose[1] * cfg.costmap.scale, 
                    start_pose[2])
    
    costmap.update(obs_vertices=obstacles,
                   obs_masses=masses,
                   ship_pos_y=start_pose[1] - ship.length / 2,
                   ship_speed=cfg.sim_dynamics.target_speed,
                   goal=goal_y * cfg.costmap.scale)
    
    # Create A* and search
    a_star = AStar(full_costmap=costmap,
                   prim=prim,
                   ship=ship,
                   swath_dict=swath_dict,
                   swath_shape=swath_shape,
                   **cfg.a_star)
    
    goal_y_scaled = goal_y * cfg.costmap.scale
    search_result = a_star.search(start=start_scaled, goal_y=goal_y_scaled)
    
    if not search_result:
        return None, None, None, None
    
    # Unpack results
    continuous_path, _, node_path_discrete, _, _, _, _, _ = search_result
    
    # Convert discrete plan to meters
    discrete_plan_meters = np.zeros((node_path_discrete.shape[1], 3))
    discrete_plan_meters[:, 0] = node_path_discrete[0] / cfg.costmap.scale
    discrete_plan_meters[:, 1] = node_path_discrete[1] / cfg.costmap.scale
    discrete_plan_meters[:, 2] = node_path_discrete[2]
    
    # Convert continuous path to meters
    continuous_path_meters = np.zeros((continuous_path.shape[1], 3))
    continuous_path_meters[:, 0] = continuous_path[0] / cfg.costmap.scale
    continuous_path_meters[:, 1] = continuous_path[1] / cfg.costmap.scale
    continuous_path_meters[:, 2] = continuous_path[2]
    
    # Fit spline to continuous path
    spline_control_points, spline_knots, spline_t = fit_spline_to_path(
        continuous_path_meters, 
        num_control_points=num_spline_control_points,
        degree=3
    )
    
    return discrete_plan_meters, continuous_path_meters, costmap, spline_control_points


def simulate_ship_tracking_path(cfg, start_pose, start_nu, continuous_path, horizon_category='medium'):
    """Simulate ship tracking the continuous path to get actual executed states"""
    # Select max_steps based on horizon category
    max_steps_map = {
        'short': MAX_SIM_STEPS_SHORT,
        'medium': MAX_SIM_STEPS_MEDIUM,
        'long': MAX_SIM_STEPS_LONG,
        'very_long': MAX_SIM_STEPS_VERY_LONG
    }
    max_steps = max_steps_map.get(horizon_category, MAX_SIM_STEPS_MEDIUM)
    
    sim_dynamics = SimShipDynamics(
        eta=start_pose, nu=start_nu,
        output_dir=None,
        **cfg.sim_dynamics
    )
    
    sim_dynamics.init_trajectory_tracking(continuous_path)
    state = sim_dynamics.state
    
    # Track states during simulation
    executed_states = []  # List of (x, y, psi, u, v, r) at each step
    executed_controls = []  # List of control inputs
    
    steps = 0
    goal_y = continuous_path[-1, 1] if len(continuous_path) > 0 else start_pose[1] + 100
    
    while steps < max_steps and state.y < goal_y:
        sim_dynamics.control()
        sim_dynamics.log_step()
        sim_dynamics.sim_step()
        state.integrate()
        
        # Record state
        executed_states.append([
            state.x, state.y, state.psi,
            state.u, state.v, state.r
        ])
        executed_controls.append(state.u_control.copy())
        
        steps += 1
        
        # Early termination if we've passed the goal
        if state.y >= goal_y:
            break
        
        # Safety check: if ship hasn't moved forward in many steps, stop
        if steps > 100 and len(executed_states) > 100:
            recent_y = [s[1] for s in executed_states[-50:]]
            if max(recent_y) - min(recent_y) < 5.0:
                # Ship stuck, stop simulation
                break
    
    executed_states = np.array(executed_states)
    executed_controls = np.array(executed_controls)
    
    return executed_states, executed_controls


def generate_dataset(num_samples=NUM_SAMPLES, output_file='neural_mpc_dataset.pkl', 
                     balance_categories=True):
    """Generate the full training dataset with long horizons and high ice density"""
    dataset = []
    
    print(f"Generating {num_samples} training samples with long horizons and high ice density...")
    print(f"Map shape: {MAP_SHAPE}")
    print(f"Goal distance range: {MIN_GOAL_DISTANCE} - {MAX_GOAL_DISTANCE} m")
    print(f"Ice concentration range: {MIN_ICE_CONCENTRATION} - {MAX_ICE_CONCENTRATION}")
    
    successful = 0
    failed = 0
    failed_planning = 0
    failed_simulation = 0
    category_counts = {'short': 0, 'medium': 0, 'long': 0, 'very_long': 0}
    concentration_stats = []
    
    # Balance samples across horizon categories
    if balance_categories:
        samples_per_category = num_samples // len(HORIZON_CATEGORIES)
        remaining = num_samples % len(HORIZON_CATEGORIES)
        category_allocations = [samples_per_category] * len(HORIZON_CATEGORIES)
        for i in range(remaining):
            category_allocations[i] += 1
    else:
        category_allocations = [num_samples] + [0] * (len(HORIZON_CATEGORIES) - 1)
    
    sample_idx = 0
    for cat_idx, (min_dist, max_dist, cat_name) in enumerate(HORIZON_CATEGORIES):
        num_cat_samples = category_allocations[cat_idx]
        print(f"\nGenerating {num_cat_samples} samples for {cat_name} horizon ({min_dist:.0f}-{max_dist:.0f}m)...")
        
        for i in tqdm(range(num_cat_samples), desc=f"{cat_name} horizon"):
            try:
                # Generate random scenario with specific horizon category
                result = generate_random_scenario(CFG, seed=sample_idx, 
                                                 horizon_category=(min_dist, max_dist, cat_name))
                start_pose, goal, obstacles, masses, obstacles_dict, category_name, concentration = result
                
                # Skip if no obstacles generated (but log it)
                if not obstacles or len(obstacles) == 0:
                    print(f"\nWarning: Sample {sample_idx} - No obstacles generated, skipping...")
                    failed += 1
                    sample_idx += 1
                    continue
                
                # Get discrete and continuous plan, and spline parameters
                result = get_discrete_and_continuous_plan(
                    CFG, obstacles, masses, start_pose, goal[1], num_spline_control_points=15
                )
                
                if result is None or len(result) < 3:
                    failed_planning += 1
                    failed += 1
                    sample_idx += 1
                    continue
                
                discrete_plan, continuous_path, costmap, spline_control_points = result
                
                if discrete_plan is None or continuous_path is None or len(discrete_plan) == 0:
                    failed_planning += 1
                    failed += 1
                    sample_idx += 1
                    continue
                
                if spline_control_points is None or len(spline_control_points) == 0:
                    failed_planning += 1
                    failed += 1
                    sample_idx += 1
                    continue
                
                # Simulate ship tracking to get executed states
                start_nu = [0.0, 0.0, 0.0]  # Start from rest
                try:
                    executed_states, executed_controls = simulate_ship_tracking_path(
                        CFG, start_pose, start_nu, continuous_path, horizon_category=category_name
                    )
                    
                    # Validate executed states
                    if len(executed_states) < 10:  # Need minimum number of states
                        failed_simulation += 1
                        failed += 1
                        sample_idx += 1
                        continue
                        
                except Exception as sim_e:
                    print(f"\nSimulation error for sample {sample_idx}: {sim_e}")
                    failed_simulation += 1
                    failed += 1
                    sample_idx += 1
                    continue
                
                # Store sample with metadata (now includes spline parameters)
                sample = {
                    'discrete_plan': discrete_plan,  # (N, 3) - waypoints from A*
                    'continuous_path': continuous_path,  # (M, 3) - smooth path from A* (for reference)
                    'spline_control_points': spline_control_points,  # (K, 2) - B-spline control points [x, y]
                    'executed_states': executed_states,  # (K, 6) - actual ship states [x, y, psi, u, v, r]
                    'executed_controls': executed_controls,  # (K, 3) - control inputs
                    'start_pose': np.array(start_pose),  # (3,) - [x, y, psi]
                    'start_nu': np.array(start_nu),  # (3,) - [u, v, r]
                    'goal': np.array(goal),  # (2,) - [x, y]
                    'obstacles': obstacles,  # List of obstacle vertices
                    'masses': masses,  # List of obstacle masses
                    'costmap': costmap.cost_map if costmap else None,  # Costmap for reference
                    'map_shape': MAP_SHAPE,
                    'dt': DT,
                    'target_speed': TARGET_SPEED,
                    'horizon_category': category_name,  # Metadata
                    'ice_concentration': concentration,  # Actual ice concentration
                    'path_length': np.linalg.norm(goal[:2] - np.array(start_pose[:2])),  # Straight-line distance
                    'num_obstacles': len(obstacles)
                }
                
                dataset.append(sample)
                successful += 1
                category_counts[category_name] += 1
                concentration_stats.append(concentration)
                sample_idx += 1
                
            except Exception as e:
                print(f"\nError generating sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                sample_idx += 1
                continue
    
    print(f"\nDataset generation complete!")
    print(f"  Successful: {successful}/{num_samples}")
    print(f"  Failed: {failed}/{num_samples}")
    print(f"    - Planning failures: {failed_planning}")
    print(f"    - Simulation failures: {failed_simulation}")
    print(f"\nHorizon category distribution:")
    for cat_name, count in category_counts.items():
        print(f"    - {cat_name}: {count} samples")
    if concentration_stats:
        print(f"\nIce concentration statistics:")
        print(f"    - Mean: {np.mean(concentration_stats):.3f}")
        print(f"    - Min: {np.min(concentration_stats):.3f}")
        print(f"    - Max: {np.max(concentration_stats):.3f}")
        print(f"    - Std: {np.std(concentration_stats):.3f}")
    
    # Save dataset
    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'dataset': dataset,
            'config': {
                'num_samples': len(dataset),
                'map_shape': MAP_SHAPE,
                'dt': DT,
                'target_speed': TARGET_SPEED,
                'vessel_model': CFG.sim_dynamics.vessel_model,
                'horizon_categories': HORIZON_CATEGORIES,
                'ice_concentration_range': [MIN_ICE_CONCENTRATION, MAX_ICE_CONCENTRATION],
                'category_distribution': category_counts
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")
    
    # Print statistics
    if len(dataset) > 0:
        discrete_lengths = [len(s['discrete_plan']) for s in dataset]
        continuous_lengths = [len(s['continuous_path']) for s in dataset]
        executed_lengths = [len(s['executed_states']) for s in dataset]
        path_lengths = [s['path_length'] for s in dataset]
        obstacle_counts = [s['num_obstacles'] for s in dataset]
        
        print(f"\nDataset Statistics:")
        print(f"  Discrete plan lengths: min={min(discrete_lengths)}, max={max(discrete_lengths)}, mean={np.mean(discrete_lengths):.1f}")
        print(f"  Continuous path lengths: min={min(continuous_lengths)}, max={max(continuous_lengths)}, mean={np.mean(continuous_lengths):.1f}")
        print(f"  Executed path lengths: min={min(executed_lengths)}, max={max(executed_lengths)}, mean={np.mean(executed_lengths):.1f}")
        print(f"  Path distances: min={min(path_lengths):.1f}m, max={max(path_lengths):.1f}m, mean={np.mean(path_lengths):.1f}m")
        print(f"  Obstacle counts: min={min(obstacle_counts)}, max={max(obstacle_counts)}, mean={np.mean(obstacle_counts):.1f}")
    
    return dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate neural MPC training dataset')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='neural_mpc_dataset.pkl',
                       help='Output pickle file path')
    
    args = parser.parse_args()
    
    generate_dataset(num_samples=args.num_samples, output_file=args.output)

