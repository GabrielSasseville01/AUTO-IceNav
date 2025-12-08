"""Static diffusion demo v2 - Extract discrete A* plan, optimize with diffusion, show all trajectories clearly
This version:
1. Directly gets discrete A* plan (node_path)
2. Uses discrete plan as initialization for diffusion
3. Shows discrete plan, baseline path, and diffusion trajectories with proper visibility
4. Includes both costmap view and actual map view
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from multiprocessing import Process, Pipe, Queue
from queue import Empty
import time

from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.utils.sim_utils import load_real_obstacles, ICE_DENSITY, ICE_THICKNESS, SHIP_MASS
from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES, Ship
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.launch import launch
from ship_ice_planner.utils.utils import DotDict
from ship_ice_planner import FULL_SCALE_SIM_PARAM_CONFIG

def get_discrete_plan_from_lattice(cfg, obstacles, masses, start_pose, goal_y, scale):
    """
    Directly call A* to get the discrete node plan (not the continuous path).
    Returns discrete plan in meters.
    """
    # Setup same as lattice planner
    ship = Ship(scale=scale, **cfg.ship)
    prim = Primitives(**cfg.prim)
    swath_dict, swath_shape = generate_swath(ship, prim)
    
    costmap = CostMap(length=cfg.map_shape[0],
                     width=cfg.map_shape[1],
                     ship_mass=ship.mass,
                     padding=swath_shape[0] // 2,
                     **cfg.costmap)
    
    # Update costmap
    costmap.update(obs_vertices=obstacles,
                   obs_masses=masses,
                   ship_pos_y=start_pose[1],
                   ship_speed=cfg.sim_dynamics.target_speed,
                   goal=goal_y)
    
    # Create A* and search
    a_star = AStar(full_costmap=costmap,
                   prim=prim,
                   ship=ship,
                   swath_dict=swath_dict,
                   swath_shape=swath_shape,
                   **cfg.a_star)
    
    # Search in costmap coordinates
    start_scaled = (start_pose[0] * scale, start_pose[1] * scale, start_pose[2])
    goal_y_scaled = goal_y * scale
    
    search_result = a_star.search(start=start_scaled, goal_y=goal_y_scaled)
    
    if not search_result:
        return None, None, None
    
    # Unpack: new_path, new_swath, node_path, nodes_expanded, g_score, swath_cost, path_length, edge_seq
    continuous_path, _, node_path_discrete, _, _, _, _, _ = search_result
    
    # Convert discrete plan to meters
    # node_path_discrete is (3, N) in costmap coordinates
    discrete_plan_meters = np.zeros((node_path_discrete.shape[1], 3))
    discrete_plan_meters[:, 0] = node_path_discrete[0] / scale  # x
    discrete_plan_meters[:, 1] = node_path_discrete[1] / scale  # y
    discrete_plan_meters[:, 2] = node_path_discrete[2]  # psi (already radians)
    
    # Convert continuous path to meters
    continuous_path_meters = np.zeros((continuous_path.shape[1], 3))
    continuous_path_meters[:, 0] = continuous_path[0] / scale
    continuous_path_meters[:, 1] = continuous_path[1] / scale
    continuous_path_meters[:, 2] = continuous_path[2]
    
    return discrete_plan_meters, continuous_path_meters, costmap

def demo_static():
    # 1. Setup Environment
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Map Configuration
    MAP_FILE = "data/MEDEA_fram_20100629.pkl"
    PADDING = 50
    
    width = 600  # m
    height = 600  # m
    scale = 0.5  # pixels per meter

    # Ship
    ship = Ship(scale=scale, vertices=FULL_SCALE_PSV_VERTICES, mass=SHIP_MASS)
    start_pose = np.array([50.0, 0.0, np.deg2rad(90)])
    start_nu = np.array([2.0, 0.0, 0.0])
    goal = [50.0, 600.0]
    
    # Load Real Obstacles
    print(f"Loading map from {MAP_FILE}...")
    obs_dicts, _ = load_real_obstacles(MAP_FILE)
    
    # Shift obstacles
    print(f"Applying padding of {PADDING}m...")
    for ob in obs_dicts:
        ob['vertices'][:, 1] += np.array(PADDING).astype(np.int32)
        ob['centre'] = np.mean(ob['vertices'], axis=0)
        
    obstacles = [ob['vertices'] for ob in obs_dicts]
    
    # Calculate masses
    masses = []
    from ship_ice_planner.geometry.polygon import poly_area
    for ob in obs_dicts:
        if 'mass' in ob:
            masses.append(ob['mass'])
        elif 'area' in ob:
            masses.append(ob['area'] * ICE_DENSITY * ICE_THICKNESS)
        else:
            area = poly_area(ob['vertices'])
            masses.append(area * ICE_DENSITY * ICE_THICKNESS)

    # Load config
    print("Loading config...")
    CFG = DotDict.load_from_file(FULL_SCALE_SIM_PARAM_CONFIG)
    CFG.map_file = MAP_FILE
    CFG.ship.start_pos = list(start_pose)
    CFG.ship.goal_pos = goal
    CFG.ship.spawn_padding = PADDING

    # 2. Get discrete plan from A* directly
    print("\n" + "=" * 60)
    print("Step 1: Getting discrete A* plan...")
    print("=" * 60)
    
    discrete_plan, continuous_path, costmap = get_discrete_plan_from_lattice(
        CFG, obstacles, masses, start_pose, goal[1], scale
    )
    
    if discrete_plan is None:
        print("ERROR: Failed to get discrete plan. Using straight line fallback.")
        num_points = 500
        discrete_plan = np.zeros((num_points, 3))
        discrete_plan[:, 0] = start_pose[0]
        discrete_plan[:, 1] = np.linspace(start_pose[1], goal[1], num_points)
        discrete_plan[:, 2] = start_pose[2]
        continuous_path = discrete_plan.copy()
    
    print(f"Discrete plan: {discrete_plan.shape} waypoints")
    print(f"Continuous path: {continuous_path.shape} waypoints")

    # 3. Run diffusion optimization using discrete plan as warm start
    print("\n" + "=" * 60)
    print("Step 2: Running diffusion optimization with discrete plan initialization...")
    print("=" * 60)
    
    num_trials = 2
    colors = ['lime', 'orange', 'cyan']
    diffusion_paths = []
    sampled_trajectories_all = []
    
    # Use FULL continuous path length for horizon - go all the way to goal in 600x600 box
    # Don't cap too aggressively - ensure we reach the full goal distance
    horizon = min(len(continuous_path), 300)  # Use continuous path, reasonable cap for performance
    # Ensure we're optimizing for the full goal distance in the bounding box
    goal_y_target = goal[1]  # Full goal distance (600m)
    
    # Create interpolated waypoint anchor path from discrete plan
    # Map discrete plan waypoints to corresponding indices in continuous path
    discrete_indices = np.linspace(0, len(continuous_path) - 1, len(discrete_plan)).astype(int)
    anchor_waypoints = discrete_plan.copy()  # Keep original discrete waypoints
    
    # Interpolate discrete plan to horizon length for local_path
    if len(continuous_path) > horizon:
        # Use first horizon steps of continuous path
        local_path = continuous_path[:horizon].copy()
    else:
        local_path = continuous_path.copy()
        horizon = len(continuous_path)
    
    # Store waypoint anchors (mapped to local_path indices)
    waypoint_anchors = {}  # {time_step: (x, y, psi)}
    for i, wp_idx in enumerate(discrete_indices):
        if wp_idx < horizon:
            waypoint_anchors[wp_idx] = discrete_plan[i]
    
    # Ensure we have waypoint anchors throughout the trajectory (not just at start)
    # Add more anchors if discrete plan is too sparse
    if len(waypoint_anchors) < 3:
        # Interpolate waypoints more evenly across horizon
        for i in range(0, horizon, max(horizon // 5, 1)):
            if i not in waypoint_anchors:
                # Find closest discrete waypoint
                closest_idx = np.argmin([abs(i - idx) for idx in discrete_indices if idx < horizon])
                waypoint_anchors[i] = discrete_plan[closest_idx]
    
    print(f"Using horizon: {horizon} steps (from continuous path)")
    print(f"Discrete plan has {len(discrete_plan)} waypoints, mapping to {len(waypoint_anchors)} anchors")
    print(f"Running {num_trials} diffusion trials...")
    
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}...")
        
        controller = ModelBasedDiffusionController(
            horizon=horizon,
            num_samples=512,
            num_diffusion_steps=50,
            temperature=0.1,
            seed=seed + i
        )
        
        controller.target_surge_speed = start_nu[0]
        controller.enforce_forward_motion = True
        
        # Store waypoint anchors in controller for objective function
        controller.waypoint_anchors = waypoint_anchors
        
        # Initialize with continuous path (better warm start)
        warm_start_controls = np.zeros((horizon, 3))
        eta = np.array(start_pose, dtype=float).copy()
        current_nu = np.array(start_nu, dtype=float).copy()
        dt = 0.1
        
        from ship_ice_planner.geometry.utils import Rxy
        for t in range(horizon):
            target = local_path[t] if t < len(local_path) else local_path[-1]
            pos_err = target[:2] - eta[:2]
            ang_err = target[2] - eta[2]
            ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
            
            R = Rxy(eta[2])
            body_err = R.T @ pos_err
            
            warm_start_controls[t, 0] = 900.0  # Strong forward control
            warm_start_controls[t, 1] = np.clip(body_err[1] * 400.0, -600, 600)
            warm_start_controls[t, 2] = np.clip(ang_err * 200.0, -250, 250)
            
            # Approximate dynamics
            current_nu[0] = min(current_nu[0] + dt * 0.15, start_nu[0])
            u_g, v_g = Rxy(eta[2]) @ [current_nu[0], current_nu[1]]
            eta[0] += dt * u_g * 0.6
            eta[1] += dt * v_g * 0.6
            eta[2] = (eta[2] + dt * current_nu[2] * 0.6) % (2 * np.pi)
        
        controller.u_seq_mean = warm_start_controls
        controller.execution_count = 0
        
        # Ensure start_pose is numpy array
        start_pose_arr = np.array(start_pose, dtype=float)
        start_nu_arr = np.array(start_nu, dtype=float)
        
        # Ensure local_path extends to goal (if it doesn't already)
        if local_path[-1, 1] < goal_y_target:
            # Extend path to goal
            num_extra = int((goal_y_target - local_path[-1, 1]) / (dt * start_nu[0])) + 1
            if num_extra > 0:
                last_point = local_path[-1].copy()
                last_point[1] = goal_y_target
                extended_path = np.linspace(local_path[-1], last_point, num_extra + 1)[1:]
                local_path = np.vstack([local_path, extended_path])
                horizon = len(local_path)
                # Update waypoint anchors for extended path
                for wp_t, wp in list(waypoint_anchors.items()):
                    if wp_t >= len(local_path):
                        del waypoint_anchors[wp_t]
                controller.horizon = horizon
                controller.u_seq_mean = np.vstack([
                    controller.u_seq_mean,
                    np.tile(controller.u_seq_mean[-1:], (len(extended_path), 1))
                ])
        
        # Run diffusion - don't condition on goal separately, discrete path handles it
        controller.DPcontrol(
            pose=start_pose_arr,
            setpoint=[start_pose_arr[0], goal_y_target, start_pose_arr[2]],
            dt=dt,
            nu=start_nu_arr,
            local_path=local_path,
            costmap=costmap,
            goal_y=None  # Don't condition on goal - discrete path waypoints handle it!
        )
        
        # Store results
        if hasattr(controller, 'predicted_trajectory') and controller.predicted_trajectory is not None:
            diffusion_paths.append(controller.predicted_trajectory.copy())
            print(f"  Trial {i+1} complete: trajectory shape {controller.predicted_trajectory.shape}")
        else:
            print(f"  Trial {i+1} failed: using discrete plan")
            diffusion_paths.append(local_path.copy())
        
        if i == num_trials - 1 and hasattr(controller, 'sampled_trajectories_viz'):
            sampled_trajectories_all = controller.sampled_trajectories_viz

    # 4. Plot Comparison - TWO VIEWS: Costmap and Actual Map
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)
    
    fig = plt.figure(figsize=(20, 10))
    
    # Left subplot: Costmap view
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Plot CostMap
    cmap_img = costmap.cost_map.copy()
    if cmap_img.sum() == 0: 
        cmap_img[:] = 1 
    ax1.imshow(cmap_img, origin='lower', cmap='viridis', extent=[0, width, 0, height], alpha=0.5)
    
    # Plot Obstacles (costmap view)
    from matplotlib import patches
    for obs in costmap.obstacles:
        poly_verts = obs['vertices'] / costmap.scale
        poly = patches.Polygon(poly_verts, fill=False, edgecolor='cyan', linewidth=0.5, alpha=0.4)
        ax1.add_patch(poly)
    
    # Plot Ship
    from ship_ice_planner.geometry.utils import Rxy
    R = Rxy(start_pose[2])
    ship_poly = ship.vertices @ R.T + [start_pose[0], start_pose[1]]
    ax1.add_patch(patches.Polygon(ship_poly, True, fill=True, color='red', label='Ship Start', 
                                  zorder=10, edgecolor='black', linewidth=1.5))
    
    # Plot Goal
    ax1.scatter([goal[0]], [goal[1]], c='green', s=300, marker='*', 
               label='Goal', zorder=10, edgecolors='black', linewidths=2)
    
    # Plot Discrete Plan (with markers)
    ax1.plot(discrete_plan[:, 0], discrete_plan[:, 1], 
            'mo-', linewidth=3, markersize=8, label='Discrete A* Plan', 
            alpha=0.9, zorder=6, markevery=max(1, len(discrete_plan)//20))
    
    # Plot Continuous Path (baseline)
    ax1.plot(continuous_path[:, 0], continuous_path[:, 1], 
            'b--', linewidth=3, label='Continuous Baseline', alpha=0.8, zorder=5)
    
    # Plot Diffusion Paths (high zorder)
    for i, diff_path in enumerate(diffusion_paths):
        label = f'Diffusion Trial {i+1}' if i == 0 else None
        ax1.plot(diff_path[:, 0], diff_path[:, 1], 
                color=colors[i], linewidth=4, label=label, alpha=0.9, zorder=7+i)
    
    ax1.set_title("Costmap View: All Trajectories", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X (m)", fontsize=12)
    ax1.set_ylabel("Y (m)", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.set_xlim(0, width * 2)  # 2x bounds
    ax1.set_ylim(0, height * 2)  # 2x bounds
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right subplot: Actual Map View (like demo_sim2d)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Plot obstacles as polygons (actual map)
    for obs in obstacles:
        poly = patches.Polygon(obs, fill=True, edgecolor='gray', facecolor='lightblue', 
                              linewidth=1, alpha=0.6, zorder=1)
        ax2.add_patch(poly)
    
    # Plot Ship
    ship_poly2 = ship.vertices @ R.T + [start_pose[0], start_pose[1]]
    ax2.add_patch(patches.Polygon(ship_poly2, True, fill=True, color='red', label='Ship Start', 
                                  zorder=10, edgecolor='black', linewidth=2))
    
    # Plot Goal
    ax2.scatter([goal[0]], [goal[1]], c='green', s=400, marker='*', 
               label='Goal', zorder=10, edgecolors='black', linewidths=2)
    
    # Plot Discrete Plan
    ax2.plot(discrete_plan[:, 0], discrete_plan[:, 1], 
            'mo-', linewidth=3, markersize=10, label='Discrete A* Plan', 
            alpha=0.9, zorder=6, markevery=max(1, len(discrete_plan)//20))
    
    # Plot Continuous Baseline
    ax2.plot(continuous_path[:, 0], continuous_path[:, 1], 
            'b--', linewidth=3, label='Continuous Baseline', alpha=0.8, zorder=5)
    
    # Plot Diffusion Paths (high zorder for visibility)
    for i, diff_path in enumerate(diffusion_paths):
        label = f'Diffusion Trial {i+1}' if i == 0 else None
        ax2.plot(diff_path[:, 0], diff_path[:, 1], 
                color=colors[i], linewidth=4, label=label, alpha=0.95, zorder=8+i)
    
    # Plot sampled trajectories (last trial only)
    if sampled_trajectories_all:
        print(f"Plotting {len(sampled_trajectories_all)} sampled trajectories...")
        for j, traj in enumerate(sampled_trajectories_all[:10]):
            label = 'Diffusion Samples' if j == 0 else None
            ax2.plot(traj[:, 0], traj[:, 1], 'white', alpha=0.15, linewidth=0.8, 
                    label=label, zorder=2)
    
    ax2.set_title("Actual Map View: Trajectories on Real Obstacles", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X (m)", fontsize=12)
    ax2.set_ylabel("Y (m)", fontsize=12)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax2.set_xlim(0, width * 2)  # 2x bounds
    ax2.set_ylim(0, height * 2)  # 2x bounds
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('diffusion_static_demo_v2.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to diffusion_static_demo_v2.png")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Comparison Statistics:")
    print("=" * 60)
    discrete_length = np.sum(np.linalg.norm(np.diff(discrete_plan[:, :2], axis=0), axis=1))
    continuous_length = np.sum(np.linalg.norm(np.diff(continuous_path[:, :2], axis=0), axis=1))
    print(f"Discrete A* plan length: {discrete_length:.2f} m")
    print(f"Continuous baseline length: {continuous_length:.2f} m")
    
    for i, diff_path in enumerate(diffusion_paths):
        diff_length = np.sum(np.linalg.norm(np.diff(diff_path[:, :2], axis=0), axis=1))
        improvement = ((discrete_length - diff_length) / discrete_length * 100) if discrete_length > 0 else 0
        print(f"Diffusion Trial {i+1} length: {diff_length:.2f} m ({improvement:+.1f}% vs discrete)")
    
    plt.show()

if __name__ == "__main__":
    demo_static()

