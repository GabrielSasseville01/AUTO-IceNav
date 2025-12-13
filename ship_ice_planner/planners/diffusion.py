"""
Diffusion-based planner for ship ice navigation.
Combines A* discrete planning with model-based diffusion optimization.
"""
import logging
import os
import pickle
import time
import traceback

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ship_ice_planner import PLANNER_PLOT_DIR, PATH_DIR, METRICS_FILE
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.geometry.utils import Rxy
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import resample_path
from ship_ice_planner.utils.path_compare import Path


def optimize_with_diffusion(discrete_plan, continuous_path, start_pose, start_nu, dt, target_speed,
                            costmap, diffusion_cfg, goal_y_full=None):
    """
    Optimize path using diffusion with discrete plan as waypoint anchors.
    
    :param discrete_plan: (N, 3) discrete A* plan waypoints
    :param continuous_path: (M, 3) continuous path from A*
    :param start_pose: (3,) starting pose [x, y, psi]
    :param start_nu: (3,) starting velocities [u, v, r]
    :param dt: time step
    :param target_speed: target forward speed
    :param costmap: CostMap object
    :param diffusion_cfg: diffusion configuration dict
    :param goal_y_full: optional full goal y-coordinate
    :return: optimized path (N, 3)
    """
    logger = logging.getLogger(__name__)
    logger.info("Optimizing with diffusion from discrete plan...")
    logger.info(f"Discrete plan length: {len(discrete_plan)} waypoints")
    logger.info(f"Continuous path length: {len(continuous_path)} waypoints")
    
    # Use continuous path for horizon (discrete is too sparse)
    horizon = diffusion_cfg.get('horizon', 150)
    horizon = min(horizon, len(continuous_path))
    
    # Use continuous path for local_path (better reference)
    if len(continuous_path) > horizon:
        local_path = continuous_path[:horizon].copy()
    else:
        local_path = continuous_path.copy()
        horizon = len(continuous_path)
    
    # Determine goal_y
    if goal_y_full is not None:
        goal_y = goal_y_full
    else:
        goal_y = local_path[-1, 1]
    
    # Map discrete plan waypoints to anchor indices in local_path
    discrete_indices = np.linspace(0, horizon - 1, len(discrete_plan)).astype(int)
    waypoint_anchors = {}
    for i, wp_idx in enumerate(discrete_indices):
        if wp_idx < horizon:
            waypoint_anchors[wp_idx] = discrete_plan[i]
    
    logger.info(f"Using horizon: {horizon} steps")
    logger.info(f"Mapped {len(waypoint_anchors)} discrete waypoints as anchors")
    
    # Initialize controller
    controller = ModelBasedDiffusionController(
        horizon=horizon,
        diffusion_noise_clipping=diffusion_cfg.get('noise_clipping', 50),
        num_samples=diffusion_cfg.get('num_samples', 128),
        num_diffusion_steps=diffusion_cfg.get('num_diffusion_steps', 30),
        temperature=diffusion_cfg.get('temperature', 0.1),
        beta0=diffusion_cfg.get('beta0', 1e-4),
        betaT=diffusion_cfg.get('betaT', 1e-2),
        seed=0
    )
    
    controller.target_surge_speed = target_speed
    controller.enforce_forward_motion = True
    controller.waypoint_anchors = waypoint_anchors
    
    # Create warm start from continuous path
    warm_start_controls = np.zeros((horizon, 3))
    eta = np.array(start_pose, dtype=float)
    current_nu = np.array(start_nu, dtype=float)
    
    for t in range(horizon):
        target = local_path[t] if t < len(local_path) else local_path[-1]
        pos_err = target[:2] - eta[:2]
        ang_err = target[2] - eta[2]
        ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
        
        R = Rxy(eta[2])
        body_err = R.T @ pos_err
        
        # Control initialization
        warm_start_controls[t, 0] = 900.0  # Strong forward
        warm_start_controls[t, 1] = np.clip(body_err[1] * 400.0, -600, 600)
        warm_start_controls[t, 2] = np.clip(ang_err * 200.0, -250, 250)
        
        # Approximate dynamics update
        current_nu[0] = min(current_nu[0] + dt * 0.15, target_speed)
        u_g, v_g = Rxy(eta[2]) @ [current_nu[0], current_nu[1]]
        eta[0] += dt * u_g * 0.6
        eta[1] += dt * v_g * 0.6
        eta[2] = (eta[2] + dt * current_nu[2] * 0.6) % (2 * np.pi)
    
    controller.u_seq_mean = warm_start_controls
    controller.execution_count = 0
    
    logger.info("Running diffusion optimization...")
    
    # Ensure inputs are numpy arrays
    start_pose_arr = np.array(start_pose, dtype=float)
    start_nu_arr = np.array(start_nu, dtype=float)
    
    # Run diffusion optimization
    controller.DPcontrol(
        pose=start_pose_arr,
        setpoint=local_path[0] if len(local_path) > 0 else start_pose_arr,
        dt=dt,
        nu=start_nu_arr,
        local_path=local_path,
        costmap=costmap,
        goal_y=None  # Discrete path handles goal reaching
    )
    
    # Get optimized trajectory
    if hasattr(controller, 'predicted_trajectory') and controller.predicted_trajectory is not None:
        optimized_traj = controller.predicted_trajectory.copy()
        logger.info(f"Diffusion complete! Trajectory shape: {optimized_traj.shape}")
        
        # Extend trajectory if it doesn't reach the goal
        goal_y_target = local_path[-1, 1] if len(local_path) > 0 else goal_y
        final_y = optimized_traj[-1, 1]
        
        if abs(final_y - goal_y_target) > 50.0:
            logger.warning(f"Trajectory only reaches y={final_y:.1f}, extending to goal {goal_y_target:.1f}")
            num_extra = int(abs(goal_y_target - final_y) / (dt * target_speed)) + 10
            last_point = optimized_traj[-1].copy()
            goal_point = np.array([last_point[0], goal_y_target, last_point[2]])
            extended = np.linspace(last_point, goal_point, num_extra + 1)[1:]
            optimized_traj = np.vstack([optimized_traj, extended])
        
        # Extend with remaining continuous path if needed
        if len(continuous_path) > horizon:
            remaining = continuous_path[horizon:].copy()
            optimized_full = np.vstack([optimized_traj, remaining])
            return optimized_full
        return optimized_traj
    else:
        logger.warning("No trajectory from diffusion, using continuous path")
        return continuous_path[:horizon] if len(continuous_path) >= horizon else continuous_path


def diffusion_planner(cfg, debug=False, **kwargs):
    """
    Main diffusion planner function.
    Uses A* for discrete planning and diffusion for trajectory optimization.
    
    :param cfg: configuration DotDict
    :param debug: enable debug mode
    :param kwargs: additional arguments (pipe, queue for communication)
    :return: metrics history
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting up diffusion planner...')
    
    try:
        # Setup message dispatcher (same as lattice planner)
        md = get_communication_interface(**kwargs)
        md.start()
        
        # Instantiate main objects (same as lattice planner)
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(**cfg.prim)
        swath_dict, swath_shape = generate_swath(ship, prim)
        costmap = CostMap(length=cfg.map_shape[0],
                          width=cfg.map_shape[1],
                          ship_mass=ship.mass,
                          padding=swath_shape[0] // 2,
                          **cfg.costmap)
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
        
        # Get diffusion configuration (with defaults)
        diffusion_cfg = cfg.get('diffusion', {})
        if not diffusion_cfg:
            logger.debug('Using default diffusion config...')
            diffusion_cfg = {
                'horizon': 150,
                'num_samples': 128,
                'num_diffusion_steps': 30,
                'temperature': 0.1,
                'beta0': 1e-4,
                'betaT': 1e-2
            }
        
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
        analysis_save_dir = getattr(cfg, 'analysis_save_path', None)
        if analysis_save_dir:
            os.makedirs(analysis_save_dir, exist_ok=True)

        if not cfg.plot.show and plot_dir:
            matplotlib.use('Agg')
        
        # Planning state variables
        replan_count = 0
        compute_time = []
        prev_goal_y = np.inf
        ship_actual_path = ([], [])
        horizon = cfg.get('horizon', 0) * costmap.scale
        plot = None
        
        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.inf
        
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
            
            # Scale by the scaling factor
            ship_pos[:2] *= costmap.scale
            
            # Compute current goal accounting for horizon
            if goal is not None:
                prev_goal_y = goal[1] * costmap.scale
            if horizon:
                sub_goal = ship_pos[1] + horizon
                goal_y = min(prev_goal_y, sub_goal)
            else:
                goal_y = prev_goal_y
            
            # Stop planner when ship is within 1 ship length of the goal
            if ship_pos[1] >= goal_y - ship.length:
                logger.info('\033[92mAt final goal!\033[0m')
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
            
            # Convert discrete plan to meters for diffusion
            discrete_plan_meters = np.zeros((node_path.shape[1], 3))
            discrete_plan_meters[:, 0] = node_path[0] / costmap.scale
            discrete_plan_meters[:, 1] = node_path[1] / costmap.scale
            discrete_plan_meters[:, 2] = node_path[2]
            
            # Convert continuous path to meters
            continuous_path_meters = np.zeros((path_compare.path.shape[1], 3))
            continuous_path_meters[:, 0] = path_compare.path[0] / costmap.scale
            continuous_path_meters[:, 1] = path_compare.path[1] / costmap.scale
            continuous_path_meters[:, 2] = path_compare.path[2]
            
            # Step 2: Optimize with diffusion
            t = time.time()
            start_pose = (ship_pos[0] / costmap.scale, ship_pos[1] / costmap.scale, ship_pos[2])
            start_nu = [0, 0, 0]  # Initial velocities (could be improved with actual state)
            
            optimized_path = optimize_with_diffusion(
                discrete_plan=discrete_plan_meters,
                continuous_path=continuous_path_meters,
                start_pose=start_pose,
                start_nu=start_nu,
                dt=cfg.sim_dynamics.dt,
                target_speed=cfg.sim_dynamics.target_speed,
                costmap=costmap,
                diffusion_cfg=diffusion_cfg,
                goal_y_full=goal_y / costmap.scale
            )
            logger.info('Diffusion optimization time: {}'.format(time.time() - t))
            
            # The optimized path is already in meters (real world scale)
            path_real_world_scale = optimized_path
            
            compute_time.append((time.time() - t0))
            logger.info('Step time: {}'.format(compute_time[-1]))
            logger.info('Average planner rate: {} Hz\n'.format(1 / np.mean(compute_time)))
            
            # Send path to simulation
            md.send_message(resample_path(path_real_world_scale, cfg.path_step_size),
                            costmap=costmap.cost_map)
            
            t1 = time.time()
            
            # Log metrics
            metrics.put_scalars(
                iteration=replan_count,
                compute_time=compute_time[-1],
                node_cnt=node_path.shape[1],
                expanded_cnt=len(nodes_expanded),
                g_score=g_score,
                swath_cost=swath_cost,
                path_length=path_length,
                diffusion_path_length=len(optimized_path)
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
                            'diffusion_path': optimized_path,
                            'raw_message': md.raw_message,
                            'processed_message': md.processed_message
                            }
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if cfg.plot.show or plot_dir:
                ship_actual_path[0].append(ship_pos[0])
                ship_actual_path[1].append(ship_pos[1])
                
                # Convert optimized path to costmap scale for plotting
                optimized_path_scaled = optimized_path.copy()
                optimized_path_scaled[:, :2] *= costmap.scale
                
                if plot is None:
                    # Get aligned ridge costmap for visualization
                    aligned_ridge = costmap.get_aligned_ridge_costmap()
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
                        ridge_costmap=aligned_ridge,  # Add aligned ridge density map for separate visualization
                    )
                    plot = Plot(
                        **plot_args,
                        path=optimized_path_scaled.T,
                        horizon=horizon,
                        global_path=path_compare.path,
                        trajectory_savepath=analysis_save_dir
                    )
                else:
                    if cfg.plot.show:
                        plt.pause(0.1)
                    else:
                        plt.draw()
                    
                    plot.update_path(optimized_path_scaled.T,
                                     path_compare.swath,
                                     global_path=path_compare.path,
                                     nodes_expanded=nodes_expanded,
                                     ship_state=ship_actual_path,
                                     )
                    plot.update_obstacles(costmap.all_obstacles)
                    plot.update_ship(ship.vertices, *ship_pos)
                    # Update costmap (which already includes ridge costs from _apply_ridge_costmap)
                    # Update costmap and aligned ridge costmap for visualization
                    aligned_ridge = costmap.get_aligned_ridge_costmap()
                    plot.update_map(costmap.cost_map, ridge_costmap=aligned_ridge)
                    plot.animate_map(suffix=replan_count)
                
                if debug:
                    plt.show()
            
            replan_count += 1
            logger.info('Logging and plotting time: {}'.format(time.time() - t1))
    
    except Exception as e:
        logger.error('\033[91mException raised: {}\n{}\033[0m'.format(e, traceback.format_exc()))

    finally:
        if plot is not None:
            plot.close()
        md.close()
        return metrics.get_history()  # returns a list of scalars/metrics for each planning step
