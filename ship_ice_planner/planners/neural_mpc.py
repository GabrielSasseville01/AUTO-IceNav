"""
Neural MPC-based planner for ship ice navigation.
Combines A* discrete planning with neural model predictive control optimization.
"""
import logging
import os
import pickle
import time
import traceback
from pathlib import Path as PathLib

import matplotlib
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR, METRICS_FILE
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.neural_mpc import NeuralMPCController
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import resample_path
from ship_ice_planner.utils.path_compare import Path


def smooth_path(path, smoothing_window=5):
    """
    Smooth a path using a simple moving average to remove discontinuities.
    
    :param path: (N, 3) path array [x, y, psi]
    :param smoothing_window: window size for smoothing (must be odd)
    :return: smoothed path (N, 3)
    """
    if smoothing_window is None or smoothing_window < 2:
        return path
    
    # Ensure window is odd and at least 3
    smoothing_window = max(3, smoothing_window)
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    
    if len(path) <= smoothing_window:
        smoothing_window = len(path) if len(path) % 2 == 1 else len(path) - 1
        if smoothing_window < 3:
            return path
    
    half_window = smoothing_window // 2
    smoothed_path = path.copy()
    
    # Preserve the first/last half_window points to avoid introducing jumps at replans
    start_anchor = path[:half_window].copy()
    end_anchor = path[-half_window:].copy() if half_window > 0 else np.empty((0, path.shape[1]))
    
    # Smooth interior points only
    for i in range(half_window, len(path) - half_window):
        window = path[i - half_window:i + half_window + 1]
        smoothed_path[i, 0] = np.mean(window[:, 0])
        smoothed_path[i, 1] = np.mean(window[:, 1])
        
        if path.shape[1] == 3:
            angles = window[:, 2]
            smoothed_path[i, 2] = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    
    if half_window > 0:
        smoothed_path[:half_window] = start_anchor
        smoothed_path[-half_window:] = end_anchor
    
    return smoothed_path


def densify_discrete_plan(discrete_plan, step_size=5.0, min_points=20, max_points=100):
    """
    Densify discrete plan by interpolating to match training data density.
    """
    if len(discrete_plan) < 2:
        return discrete_plan
    
    # Calculate cumulative distance along path
    positions = discrete_plan[:, :2]
    distances = np.zeros(len(discrete_plan))
    for i in range(1, len(discrete_plan)):
        distances[i] = distances[i-1] + np.linalg.norm(positions[i] - positions[i-1])
    
    total_distance = distances[-1]
    
    # Determine number of points based on step_size
    num_points = int(total_distance / step_size)
    num_points = max(min_points, min(max_points, num_points))
    
    # If input is already dense enough, return it (or downsample)
    if len(discrete_plan) >= num_points:
        if len(discrete_plan) > max_points:
            indices = np.linspace(0, len(discrete_plan) - 1, max_points, dtype=int)
            return discrete_plan[indices]
        return discrete_plan
    
    # Create new distance array with uniform spacing
    new_distances = np.linspace(0, total_distance, num_points)
    
    # Use cubic interpolation for smoother curves (if enough points), otherwise linear
    if len(discrete_plan) >= 4:
        interp_kind = 'cubic'
    else:
        interp_kind = 'linear'
    
    # Interpolate x, y positions
    interp_x = interp1d(distances, discrete_plan[:, 0], kind=interp_kind, 
                       fill_value='extrapolate', assume_sorted=True)
    interp_y = interp1d(distances, discrete_plan[:, 1], kind=interp_kind, 
                       fill_value='extrapolate', assume_sorted=True)
    
    # Handle heading (psi) - use angle interpolation with wrapping
    psi = discrete_plan[:, 2]
    psi_unwrapped = np.unwrap(psi)
    interp_psi = interp1d(distances, psi_unwrapped, kind=interp_kind, 
                         fill_value='extrapolate', assume_sorted=True)
    
    # Generate interpolated points
    densified_x = interp_x(new_distances)
    densified_y = interp_y(new_distances)
    densified_psi = interp_psi(new_distances)
    densified_psi = (densified_psi + np.pi) % (2 * np.pi) - np.pi
    
    densified_plan = np.column_stack([densified_x, densified_y, densified_psi])
    
    return densified_plan


def optimize_with_neural_mpc(neural_mpc, discrete_plan, start_pose, start_nu, goal, costmap, nmpc_cfg):
    """
    Optimize path using Neural MPC with densified discrete plan.
    
    :param neural_mpc: NeuralMPCController instance
    :param discrete_plan: (N, 3) discrete A* plan waypoints
    :param start_pose: (3,) starting pose [x, y, psi]
    :param start_nu: (3,) starting velocities [u, v, r]
    :param goal: (2,) goal position [x, y]
    :param costmap: CostMap object
    :param nmpc_cfg: neural MPC configuration dict
    :return: optimized path (M, 3)
    """
    logger = logging.getLogger(__name__)
    logger.info("Optimizing with Neural MPC from discrete plan...")
    logger.info(f"Discrete plan length: {len(discrete_plan)} waypoints")
    
    # Plan with Neural MPC
    use_grad_opt = nmpc_cfg.get('use_gradient_optimization', True)
    
    if use_grad_opt and costmap is not None:
        optimized_path = neural_mpc.optimize_path_with_costmap(
            discrete_plan=discrete_plan,
            start_pose=np.array(start_pose, dtype=float),
            start_nu=np.array(start_nu, dtype=float),
            goal=np.array(goal, dtype=float),
            costmap=costmap
        )
    else:
        optimized_path = neural_mpc.predict_path(
            discrete_plan=discrete_plan,
            start_pose=np.array(start_pose, dtype=float),
            start_nu=np.array(start_nu, dtype=float),
            goal=np.array(goal, dtype=float)
        )
    
    logger.info(f"Neural MPC complete! Path shape: {optimized_path.shape}")
    
    # Validate and fix path to ensure forward motion
    start_y = start_pose[1]
    final_y = optimized_path[-1, 1] if len(optimized_path) > 0 else start_y
    goal_y = goal[1]
    
    # Check if path actually moves forward (at least 20m forward)
    path_distance = final_y - start_y
    total_path_length = np.sum(np.linalg.norm(np.diff(optimized_path[:, :2], axis=0), axis=1)) if len(optimized_path) > 1 else 0.0
    
    logger.info(f"Path stats: start_y={start_y:.1f}, final_y={final_y:.1f}, goal_y={goal_y:.1f}, forward_distance={path_distance:.1f}m, total_length={total_path_length:.1f}m")
    
    # If path doesn't move forward enough or is too short, extend it significantly
    min_forward_distance = 50.0  # Minimum 50m forward
    min_total_length = 100.0  # Minimum 100m total path length
    
    if path_distance < min_forward_distance or total_path_length < min_total_length or final_y < goal_y - 10.0:
        logger.warning(f"Path too short or doesn't reach goal. Extending path...")
        
        # Get current endpoint (or use start if path is bad)
        if len(optimized_path) > 0 and path_distance > 5.0:
            last_point = optimized_path[-1].copy()
            current_heading = optimized_path[-1, 2] if len(optimized_path) > 0 else start_pose[2]
        else:
            # Path didn't move much, start from actual start
            last_point = np.array([start_pose[0], start_pose[1], start_pose[2]])
            current_heading = start_pose[2]
            # If we're using start, we need to regenerate or extend more aggressively
            # Use the discrete plan endpoint as guidance
            if len(discrete_plan) > 1:
                discrete_end = discrete_plan[-1]
                current_heading = np.arctan2(discrete_end[1] - start_pose[1], discrete_end[0] - start_pose[0])
    """
    Optimize path using Neural MPC with densified discrete plan.
    
    :param neural_mpc: NeuralMPCController instance
    :param discrete_plan: (N, 3) discrete A* plan waypoints
    :param start_pose: (3,) starting pose [x, y, psi]
    :param start_nu: (3,) starting velocities [u, v, r]
    :param goal: (2,) goal position [x, y]
    :param costmap: CostMap object
    :param nmpc_cfg: neural MPC configuration dict
    :return: optimized path (M, 3)
    """
    logger = logging.getLogger(__name__)
    logger.info("Optimizing with Neural MPC from discrete plan...")
    logger.info(f"Discrete plan length: {len(discrete_plan)} waypoints")
    
    # Plan with Neural MPC
    use_grad_opt = nmpc_cfg.get('use_gradient_optimization', True)
    
    if use_grad_opt and costmap is not None:
        optimized_path = neural_mpc.optimize_path_with_costmap(
            discrete_plan=discrete_plan,
            start_pose=np.array(start_pose, dtype=float),
            start_nu=np.array(start_nu, dtype=float),
            goal=np.array(goal, dtype=float),
            costmap=costmap
        )
    else:
        optimized_path = neural_mpc.predict_path(
            discrete_plan=discrete_plan,
            start_pose=np.array(start_pose, dtype=float),
            start_nu=np.array(start_nu, dtype=float),
            goal=np.array(goal, dtype=float)
        )
    
    logger.info(f"Neural MPC complete! Path shape: {optimized_path.shape}")
    
    # Validate and fix path to ensure forward motion
    start_y = start_pose[1]
    final_y = optimized_path[-1, 1] if len(optimized_path) > 0 else start_y
    goal_y = goal[1]
    
    # Check if path actually moves forward (at least 20m forward)
    path_distance = final_y - start_y
    total_path_length = np.sum(np.linalg.norm(np.diff(optimized_path[:, :2], axis=0), axis=1)) if len(optimized_path) > 1 else 0.0
    
    logger.info(f"Path stats: start_y={start_y:.1f}, final_y={final_y:.1f}, goal_y={goal_y:.1f}, forward_distance={path_distance:.1f}m, total_length={total_path_length:.1f}m")
    
    # If path doesn't move forward enough or is too short, extend it significantly
    min_forward_distance = 50.0  # Minimum 50m forward
    min_total_length = 100.0  # Minimum 100m total path length
    
    if path_distance < min_forward_distance or total_path_length < min_total_length or final_y < goal_y - 10.0:
        logger.warning(f"Path too short or doesn't reach goal. Extending path...")
        
        # Get current endpoint (or use start if path is bad)
        if len(optimized_path) > 0 and path_distance > 5.0:
            last_point = optimized_path[-1].copy()
            current_heading = optimized_path[-1, 2] if len(optimized_path) > 0 else start_pose[2]
        else:
            # Path didn't move much, start from actual start
            last_point = np.array([start_pose[0], start_pose[1], start_pose[2]])
            current_heading = start_pose[2]
            # If we're using start, we need to regenerate or extend more aggressively
            # Use the discrete plan endpoint as guidance
            if len(discrete_plan) > 1:
                discrete_end = discrete_plan[-1]
                current_heading = np.arctan2(discrete_end[1] - start_pose[1], discrete_end[0] - start_pose[0])
        
        # Create extension that goes toward goal
        goal_point = np.array([goal[0], goal_y, current_heading])
        
        # Calculate extension needed
        dist_to_goal = np.linalg.norm(goal_point[:2] - last_point[:2])
        target_length = max(min_total_length, dist_to_goal, min_forward_distance)
        
        # Generate enough points to reach goal or minimum length
        dt_estimate = 0.02  # Estimate time step
        target_speed = 2.0  # m/s
        points_per_meter = 1.0 / (target_speed * dt_estimate)  # ~25 points per meter
        num_extra = max(50, int(target_length * points_per_meter))  # At least 50 points
        
        # Interpolate linearly toward goal
        extended = np.zeros((num_extra, 3))
        for i in range(num_extra):
            alpha = (i + 1) / num_extra
            extended[i, 0] = last_point[0] + alpha * (goal_point[0] - last_point[0])
            extended[i, 1] = last_point[1] + alpha * (goal_point[1] - last_point[1])
            # Smooth heading change
            heading_diff = goal_point[2] - current_heading
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
            extended[i, 2] = (current_heading + alpha * heading_diff) % (2 * np.pi)
        
        # Combine paths
        if len(optimized_path) > 0 and path_distance > 5.0:
            optimized_path = np.vstack([optimized_path, extended])
        else:
            # Original path was bad, replace with extension from start
            optimized_path = np.vstack([last_point.reshape(1, -1), extended])
        
        logger.info(f"Extended path: new shape={optimized_path.shape}, new_end_y={optimized_path[-1, 1]:.1f}")
    
    # Final validation: ensure path progresses forward
    final_forward_dist = optimized_path[-1, 1] - start_y
    if final_forward_dist < 10.0:
        logger.error(f"Path still doesn't move forward! final_forward_dist={final_forward_dist:.1f}m")
        # Last resort: create a simple forward path
        num_points = 200
        optimized_path = np.zeros((num_points, 3))
        optimized_path[:, 0] = start_pose[0]
        optimized_path[:, 1] = np.linspace(start_pose[1], goal_y, num_points)
        optimized_path[:, 2] = np.arctan2(goal_y - start_pose[1], goal[0] - start_pose[0])
    
    return optimized_path


def neural_mpc_planner(cfg, debug=False, **kwargs):
    """
    Main neural MPC planner function.
    Uses A* for discrete planning and neural MPC for trajectory optimization.
    
    :param cfg: configuration DotDict
    :param debug: enable debug mode
    :param kwargs: additional arguments (pipe, queue for communication)
    :return: metrics history
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting up Neural MPC planner...')
    
    try:
        # Setup message dispatcher (same as other planners)
        md = get_communication_interface(**kwargs)
        md.start()
        
        # Get neural MPC configuration
        nmpc_cfg = cfg.get('neural_mpc', {})
        model_path = nmpc_cfg.get('nmpc_model_weights', 'data/nmpc_baseline.pth')
        
        if not PathLib(model_path).exists():
            logger.error(f'Neural MPC model not found at {model_path}!')
            logger.error('Please train the model first using train_neural_mpc.py')
            raise FileNotFoundError(f'Model not found: {model_path}')
        
        # Initialize Neural MPC controller
        logger.info(f'Loading Neural MPC model from {model_path}...')
        
        # Instantiate main objects (same as other planners) - needed for costmap before creating controller
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(**cfg.prim)
        swath_dict, swath_shape = generate_swath(ship, prim)
        costmap = CostMap(length=cfg.map_shape[0],
                          width=cfg.map_shape[1],
                          ship_mass=ship.mass,
                          padding=swath_shape[0] // 2,
                          **cfg.costmap)
        
        # Create Neural MPC controller (with critic if specified)
        critic_path = nmpc_cfg.get('critic_path', None) if nmpc_cfg.get('use_critic', False) else None
        neural_mpc = NeuralMPCController(
            model_path=model_path,
            horizon=nmpc_cfg.get('horizon', 200),
            use_gradient_optimization=nmpc_cfg.get('use_gradient_optimization', True),
            optimization_steps=nmpc_cfg.get('optimization_steps', 10),
            learning_rate=nmpc_cfg.get('learning_rate', 0.01),
            max_discrete_nodes=nmpc_cfg.get('max_discrete_nodes', 50),
            num_spline_control_points=nmpc_cfg.get('num_spline_control_points', 15),
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
            critic_path=critic_path,
            costmap=costmap if critic_path else None
        )
        logger.info('Neural MPC model loaded successfully')
        a_star = AStar(full_costmap=costmap,
                       prim=prim,
                       ship=ship,
                       swath_dict=swath_dict,
                       swath_shape=swath_shape,
                       **cfg.a_star)
        path_compare = Path(threshold_path_progress=cfg.get('threshold_path_progress', 0),
                            threshold_dist_to_path=cfg.get('threshold_dist_to_path', 0) * costmap.scale,
                            threshold_cost_diff=cfg.get('threshold_cost_diff', 0),
                            ship_vertices=ship.vertices,
                            costmap=costmap)
        metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)
        
        if debug:
            logger.debug('Showing debug plots for ship footprint, primitives, costmap, and swaths...')
            ship.plot(prim.turning_radius,
                      title='Turning radius {} m\nShip dimensions {} x {} m'.format(
                          prim.turning_radius / costmap.scale,
                          ship.length / costmap.scale, ship.width / costmap.scale)
                      )
            prim.plot()
            costmap.plot(ship_pos=[(costmap.shape[1] - 1) / 2, 0, np.pi / 2],
                         ship_vertices=ship.vertices,
                         prim=prim)
        
        # Directory to store plots
        plot_dir = os.path.join(cfg.output_dir, PLANNER_PLOT_DIR) if cfg.output_dir else None
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        # Directory to store generated path at each iteration
        path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        if not cfg.plot.show and plot_dir:
            matplotlib.use('Agg')
        
        # Optional directory for analysis artifacts (costmap/polygon backgrounds, path lines, etc.)
        analysis_save_dir = getattr(cfg, 'analysis_save_path', None)
        if analysis_save_dir:
            os.makedirs(analysis_save_dir, exist_ok=True)

        # Planning state variables
        replan_count = 0
        compute_time = []
        prev_goal_y = np.inf
        ship_actual_path = ([], [])
        horizon = cfg.get('horizon', 0) * costmap.scale
        plot = None
        
        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.infty
        
        # Main planner loop
        while replan_count < max_replan:
            logger.info('Re-planning count: {}'.format(replan_count))
            
            # Get new state data
            md.receive_message()
            
            # Start timer
            t0 = time.time()
            
            # Check for shutdown signal
            if md.shutdown:
                logger.info('\033[93mReceived shutdown signal!\033[0m')
                break
            
            ship_pos = md.ship_state
            goal = md.goal
            obs = md.obstacles
            obs_masses = md.masses
            metadata = getattr(md, 'metadata', None)
            
            # Scale by the scaling factor
            ship_pos[:2] *= costmap.scale
            
            # Compute current goal accounting for horizon (receding horizon planning)
            # Get final goal from metadata or config
            final_goal = None
            if metadata is not None:
                final_goal = metadata.get('final_goal')
            if final_goal is None and hasattr(cfg, 'final_goal'):
                final_goal = cfg.final_goal
            
            if goal is not None:
                prev_goal_y = goal[1] * costmap.scale
                
                # If we have a final goal, use horizon-based sub-goals
                if final_goal is not None and horizon:
                    final_goal_y_scaled = final_goal[1] * costmap.scale
                    sub_goal_y_scaled = ship_pos[1] + horizon
                    goal_y = min(final_goal_y_scaled, sub_goal_y_scaled)
                    logger.info(f'Receding horizon: ship_y={ship_pos[1]/costmap.scale:.1f}m, '
                               f'sub_goal_y={goal_y/costmap.scale:.1f}m, '
                               f'final_goal_y={final_goal_y_scaled/costmap.scale:.1f}m')
                elif horizon:
                    sub_goal = ship_pos[1] + horizon
                    goal_y = min(prev_goal_y, sub_goal)
                else:
                    goal_y = prev_goal_y
            else:
                goal_y = prev_goal_y if goal is not None else ship_pos[1] + horizon
            
            # Stop planner when ship reaches final goal (or current sub-goal if no final goal)
            check_goal_y = final_goal[1] * costmap.scale if final_goal is not None else goal_y
            if ship_pos[1] >= check_goal_y - ship.length:
                if final_goal is not None and ship_pos[1] >= final_goal[1] * costmap.scale - ship.length:
                    logger.info('\033[92mReached final goal!\033[0m')
                else:
                    logger.info('\033[92mReached sub-goal, will replan if needed!\033[0m')
                # Continue planning if not at final goal
                if final_goal is None or ship_pos[1] < final_goal[1] * costmap.scale - ship.length:
                    replan_count += 1
                    continue
                else:
                    break
            
            # Check if there is new obstacle information
            if obs is not None:
                t = time.time()
                costmap.update(obs_vertices=obs,
                               obs_masses=obs_masses,
                               ship_pos_y=ship_pos[1] - ship.length,
                               ship_speed=cfg.sim_dynamics.target_speed,
                               goal=goal_y)
                logger.info('Costmap update time: {}'.format(time.time() - t))
            
            if debug:
                logger.debug('Showing debug plot for costmap...')
                costmap.plot(ship_pos=ship_pos,
                             ship_vertices=ship.vertices,
                             prim=prim,
                             goal=goal_y)
            
            # Step 1: Compute discrete A* path
            t = time.time()
            search_result = a_star.search(
                start=tuple(ship_pos),
                goal_y=goal_y,
            )
            logger.info('A* search time: {}'.format(time.time() - t))
            
            if not search_result:
                logger.error('\033[93mPlanner failed to find a path!\033[0m')
                if path_dir:
                    with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                        pickle.dump(
                            {**a_star.diagnostics,
                             'replan_count': replan_count,
                             'stamp': t0,
                             'goal': goal_y,
                             'ship_state': ship_pos,
                             'obstacles': costmap.obstacles,
                             'costmap': costmap.cost_map,
                             'raw_message': md.raw_message,
                             'processed_message': md.processed_message
                             },
                            handle, protocol=pickle.HIGHEST_PROTOCOL
                        )
                replan_count += 1
                continue
            
            # Unpack A* result
            new_path, new_swath, node_path, nodes_expanded, g_score, swath_cost, path_length, edge_seq = search_result
            
            # Compare new path to prev path
            path_compare.update(new_path, new_swath, ship_pos)
            
            # Convert discrete plan to meters
            discrete_plan_meters = np.zeros((node_path.shape[1], 3))
            discrete_plan_meters[:, 0] = node_path[0] / costmap.scale
            discrete_plan_meters[:, 1] = node_path[1] / costmap.scale
            discrete_plan_meters[:, 2] = node_path[2]
            
            # Convert continuous path to meters (for reference, not used by neural MPC)
            continuous_path_meters = np.zeros((path_compare.path.shape[1], 3))
            continuous_path_meters[:, 0] = path_compare.path[0] / costmap.scale
            continuous_path_meters[:, 1] = path_compare.path[1] / costmap.scale
            continuous_path_meters[:, 2] = path_compare.path[2]
            
            # Densify discrete plan to match training data
            step_size = nmpc_cfg.get('discrete_densify_step_size', 5.0)
            min_points = nmpc_cfg.get('min_discrete_points', 20)
            max_points = nmpc_cfg.get('max_discrete_points', 100)
            
            discrete_plan_densified = densify_discrete_plan(
                discrete_plan_meters, 
                step_size=step_size,
                min_points=min_points,
                max_points=max_points
            )
            logger.info(f'Densified discrete plan: {len(discrete_plan_meters)} -> {len(discrete_plan_densified)} waypoints')
            
            # Step 2: Optimize with Neural MPC
            t = time.time()
            start_pose = (ship_pos[0] / costmap.scale, ship_pos[1] / costmap.scale, ship_pos[2])
            start_nu = [0, 0, 0]  # Initial velocities (could be improved with actual state)
            goal_meters = [goal[0] if goal is not None else ship_pos[0] / costmap.scale,
                          goal_y / costmap.scale]
            
            optimized_path = optimize_with_neural_mpc(
                neural_mpc=neural_mpc,
                discrete_plan=discrete_plan_densified,
                start_pose=start_pose,
                start_nu=start_nu,
                goal=goal_meters,
                costmap=costmap,
                nmpc_cfg=nmpc_cfg
            )
            logger.info('Neural MPC optimization time: {}'.format(time.time() - t))
            
            # The optimized path is already in meters (real world scale)
            path_real_world_scale = optimized_path
            
            compute_time.append((time.time() - t0))
            logger.info('Step time: {}'.format(compute_time[-1]))
            logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))
            
            # Ensure path is smooth and continuous before sending
            # First, smooth the path to remove any discontinuities (with configurable window)
            smoothing_window = int(nmpc_cfg.get('path_smoothing_window', 5)) if nmpc_cfg else 5
            if smoothing_window and smoothing_window > 1:
                path_real_world_scale = smooth_path(path_real_world_scale, smoothing_window=smoothing_window)
            
            # Then resample to desired density
            planner_metrics = {}
            if g_score is not None:
                planner_metrics['a_star_cost'] = float(g_score)
            if getattr(neural_mpc, 'last_objective_value', None) is not None:
                planner_metrics['nmpc_objective'] = float(neural_mpc.last_objective_value)
            if not planner_metrics:
                planner_metrics = None

            md.send_message(resample_path(path_real_world_scale, cfg.path_step_size),
                            costmap=costmap.cost_map,
                            metrics=planner_metrics)
            
            # Log metrics
            metrics.put_scalars(
                iteration=replan_count,
                compute_time=compute_time[-1],
                node_cnt=node_path.shape[1],
                expanded_cnt=len(nodes_expanded),
                g_score=g_score,
                swath_cost=swath_cost,
                path_length=path_length,
                neural_mpc_path_length=len(optimized_path),
                discrete_densified_cnt=len(discrete_plan_densified)
            )
            
            logger.info(metrics.data)
            metrics.step()
            
            if path_dir and cfg.get('save_paths'):
                with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                    data = {'replan_count': replan_count,
                            'stamp': t0,
                            'goal': goal_y,
                            'ship_state': ship_pos,
                            'obstacles': costmap.obstacles,
                            'costmap': costmap.cost_map,
                            'new_path': new_path,
                            'node_path': node_path,
                            'edge_seq': edge_seq,
                            'expanded': nodes_expanded,
                            'path': path_compare.path,
                            'discrete_plan_densified': discrete_plan_densified,
                            'neural_mpc_path': optimized_path,
                            'raw_message': md.raw_message,
                            'processed_message': md.processed_message
                            }
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            replan_count += 1
            
            # Visualization of planner state (costmap + polygon map)
            if cfg.plot.show or plot_dir:
                ship_actual_path[0].append(ship_pos[0])
                ship_actual_path[1].append(ship_pos[1])

                optimized_path_scaled = np.copy(path_real_world_scale)
                if optimized_path_scaled.ndim == 2 and optimized_path_scaled.shape[1] >= 2:
                    optimized_path_scaled[:, 0] *= costmap.scale
                    optimized_path_scaled[:, 1] *= costmap.scale
                else:
                    optimized_path_scaled = optimized_path_scaled * costmap.scale

                path_for_plot = optimized_path_scaled.T if optimized_path_scaled.ndim == 2 else optimized_path_scaled

                aligned_ridge = getattr(costmap, 'get_aligned_ridge_costmap', lambda: None)()

                if plot is None:
                    plot_args = dict(
                        costmap=costmap.cost_map,
                        obstacles=costmap.all_obstacles,
                        ship_vertices=ship.vertices,
                        ship_pos=ship_pos,
                        nodes_expanded=nodes_expanded,
                        sim_figsize=None,
                        scale=costmap.scale,
                        y_axis_limit=cfg.map_shape[0] * costmap.scale,
                        save_fig_dir=plot_dir,
                        show=cfg.plot.show,
                        swath=path_compare.swath,
                        ridge_costmap=aligned_ridge,
                        path=path_for_plot,
                        horizon=horizon,
                        global_path=path_compare.path,
                    )
                    plot = Plot(**plot_args, trajectory_savepath=analysis_save_dir)
                else:
                    plot.update_path(
                        path_for_plot,
                        path_compare.swath,
                        global_path=path_compare.path,
                        nodes_expanded=nodes_expanded,
                        ship_state=ship_actual_path,
                    )
                    plot.update_obstacles(costmap.all_obstacles)
                    plot.update_ship(ship.vertices, *ship_pos)
                    plot.update_map(costmap.cost_map, ridge_costmap=aligned_ridge)
                    plot.animate_map(suffix=replan_count)
        
        logger.info('Neural MPC planner finished')
        logger.info('Total replans: {}'.format(replan_count))
        logger.info('Average compute time: {} s'.format(np.mean(compute_time) if compute_time else 0))

        if plot is not None:
            plot.close()

        return metrics.get_history()
        
    except Exception as e:
        logger.error('Error in Neural MPC planner: {}'.format(e))
        logger.error(traceback.format_exc())
        raise
