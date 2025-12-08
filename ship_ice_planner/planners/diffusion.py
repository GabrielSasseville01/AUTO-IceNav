"""
Diffusion-based planner using model-based diffusion controller
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
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.ship import Ship
from ship_ice_planner.utils.message_dispatcher import get_communication_interface
from ship_ice_planner.utils.storage import Storage
from ship_ice_planner.utils.utils import resample_path
from ship_ice_planner.geometry.utils import Rxy


def diffusion_planner(cfg, debug=False, **kwargs):
    """
    Diffusion-based planner using model-based diffusion controller.
    Similar interface to lattice_planner but uses diffusion for path generation.
    """
    # Ensure Rxy is available in function scope
    from ship_ice_planner.geometry.utils import Rxy
    
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('Starting up diffusion planner...')

    try:
        # setup message dispatcher
        md = get_communication_interface(**kwargs)
        md.start()

        # instantiate main objects
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)
        prim = Primitives(**cfg.prim)
        swath_dict, swath_shape = generate_swath(ship, prim)
        costmap = CostMap(length=cfg.map_shape[0],
                         width=cfg.map_shape[1],
                         ship_mass=ship.mass,
                         padding=swath_shape[0] // 2,  # Use swath padding for lattice planner
                         **cfg.costmap)
        
        # Initialize A* for warm start
        a_star = AStar(full_costmap=costmap,
                      prim=prim,
                      ship=ship,
                      swath_dict=swath_dict,
                      swath_shape=swath_shape,
                      **cfg.a_star)
        
        # Initialize diffusion controller with reduced steps for faster planning
        diffusion_cfg = cfg.get('diffusion', {})
        controller = ModelBasedDiffusionController(
            horizon=diffusion_cfg.get('horizon', 30),  # Reduced default
            num_samples=diffusion_cfg.get('num_samples', 256),  # Reduced default
            num_diffusion_steps=diffusion_cfg.get('num_diffusion_steps', 20),  # Reduced default for speed
            temperature=diffusion_cfg.get('temperature', 0.1),
            beta0=diffusion_cfg.get('beta0', 1e-4),
            betaT=diffusion_cfg.get('betaT', 1e-2),
            seed=cfg.get('seed', 0)
        )
        
        metrics = Storage(output_dir=cfg.output_dir, file=METRICS_FILE)

        # directory to store plots
        plot_dir = os.path.join(cfg.output_dir, PLANNER_PLOT_DIR) if cfg.output_dir else None
        if plot_dir:
            os.makedirs(plot_dir)
        # directory to store generated path at each iteration
        path_dir = os.path.join(cfg.output_dir, PATH_DIR) if cfg.output_dir else None
        if path_dir:
            os.makedirs(path_dir)
        if not cfg.plot.show and plot_dir:
            matplotlib.use('Agg')

        # keep track of the planning count
        replan_count = 0
        # keep track of planner rate
        compute_time = []
        prev_goal_y = np.inf
        # keep track of ship xy position in a list for plotting
        ship_actual_path = ([], [])
        horizon = cfg.get('horizon', 0) * costmap.scale
        plot = None

        max_replan = cfg.get('max_replan')
        max_replan = max_replan if max_replan is not None else np.infty

        # start main planner loop
        while replan_count < max_replan:
            logger.info('Re-planning count: {}'.format(replan_count))

            # get new state data
            md.receive_message()

            # start timer
            t0 = time.time()

            # check if the shutdown flag is set
            if md.shutdown:
                logger.info('\033[93mReceived shutdown signal!\033[0m')
                break

            ship_pos = md.ship_state
            goal = md.goal
            obs = md.obstacles
            obs_masses = md.masses

            # scale by the scaling factor
            ship_pos_scaled = ship_pos.copy()
            ship_pos_scaled[:2] *= costmap.scale

            # compute the current goal accounting for horizon
            if goal is not None:
                prev_goal_y = goal[1] * costmap.scale
            if horizon:
                sub_goal = ship_pos_scaled[1] + horizon
                goal_y = min(prev_goal_y, sub_goal)
            else:
                goal_y = prev_goal_y

            # stop planner when ship is within 1 ship length of the goal
            if ship_pos_scaled[1] >= goal_y - ship.length:
                logger.info('\033[92mAt final goal!\033[0m')
                break

            # check if there is new obstacle information
            if obs is not None:
                # update costmap
                t = time.time()
                costmap.update(obs_vertices=obs,
                              obs_masses=obs_masses,
                              ship_pos_y=ship_pos_scaled[1] - ship.length,
                              ship_speed=cfg.sim_dynamics.target_speed,
                              goal=goal_y)
                logger.info('Costmap update time: {}'.format(time.time() - t))

            if debug:
                logger.debug('Showing debug plot for costmap...')
                costmap.plot(ship_pos=ship_pos_scaled,
                            ship_vertices=ship.vertices,
                            goal=goal_y)

            # Step 1: Get A* path as warm start
            t = time.time()
            logger.info('Running A* for warm start...')
            search_result = a_star.search(
                start=tuple(ship_pos_scaled),  # A* expects scaled coordinates
                goal_y=goal_y,
            )
            
            warm_start_path = None
            if search_result:
                lattice_path, _, _, _, _, _, _, _ = search_result
                # Convert from costmap units to meters for warm start
                warm_start_path = np.c_[(lattice_path[:2] / costmap.scale).T, lattice_path[2]]
                logger.info('A* found warm start path with {} waypoints'.format(len(warm_start_path)))
            else:
                logger.warning('A* failed, using straight line fallback for warm start')
                # Fallback to straight line
                horizon_steps = controller.horizon
                warm_start_path = np.zeros((horizon_steps, 3))
                ship_pos_meters = ship_pos.copy()
                goal_pose = [goal[0], goal_y / costmap.scale, np.pi / 2]
                for i in range(horizon_steps):
                    alpha = i / max(horizon_steps - 1, 1)
                    warm_start_path[i, 0] = ship_pos_meters[0] * (1 - alpha) + goal_pose[0] * alpha
                    warm_start_path[i, 1] = ship_pos_meters[1] * (1 - alpha) + goal_pose[1] * alpha
                    warm_start_path[i, 2] = np.pi / 2
            
            logger.info('A* warm start time: {}'.format(time.time() - t))
            
            # Step 2: Convert warm start path to control sequence for diffusion
            t = time.time()
            ship_pos_meters = ship_pos.copy()  # Already in meters
            goal_pose = [goal[0], goal_y / costmap.scale, np.pi / 2]
            nu_start = [cfg.sim_dynamics.target_speed, 0.0, 0.0]
            
            # Resample/trim warm start path to match horizon
            horizon_steps = controller.horizon
            if len(warm_start_path) > horizon_steps:
                # Take first horizon_steps points
                local_path = warm_start_path[:horizon_steps]
            elif len(warm_start_path) < horizon_steps:
                # Repeat last point to fill horizon
                last_point = warm_start_path[-1:]
                local_path = np.vstack([warm_start_path, np.tile(last_point, (horizon_steps - len(warm_start_path), 1))])
            else:
                local_path = warm_start_path.copy()
            
            # Generate control sequence from warm start path using inverse dynamics
            # Use the controller's dynamics to generate approximate controls
            # ENFORCE CONSTANT FORWARD SPEED - only optimize sway and yaw
            target_surge = cfg.sim_dynamics.target_speed  # Maintain this forward speed
            warm_start_controls = np.zeros((horizon_steps, 3))
            eta = ship_pos_meters.copy()
            nu = np.array(nu_start)
            
            # Calculate surge control needed to maintain target speed
            # Use inverse dynamics: from dynamics model, we need to find u_control[0] that gives target_surge
            # Approximate: if current surge < target, apply positive surge control
            base_surge_control = 500.0 if target_surge > 0.1 else 0.0  # Base surge to maintain forward motion
            
            for t in range(horizon_steps):
                if t < len(warm_start_path):
                    target = warm_start_path[t]
                    # Simple PD control to reach target (approximate warm start)
                    pos_err = target[:2] - eta[:2]
                    ang_err = target[2] - eta[2]
                    ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
                    
                    # Rotate error to body frame
                    R = Rxy(eta[2])
                    body_err = R.T @ pos_err
                    
                    # Generate approximate control with CONSTANT FORWARD SPEED
                    # Surge: maintain constant forward speed (don't optimize this)
                    current_surge = nu[0]
                    surge_error = target_surge - current_surge
                    # Strong forward control - always push forward
                    surge_control = max(500.0, base_surge_control + surge_error * 2000.0)  # Strong forward control
                    
                    warm_start_controls[t] = [
                        surge_control,  # surge - maintain forward speed
                        body_err[1] * 100.0,  # sway - optimize lateral position
                        ang_err * 50.0  # yaw rate - optimize heading
                    ]
                    
                    # Simulate one step to get next state for warm start
                    nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
                    nu_next_deg = controller.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], warm_start_controls[t])
                    nu = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
                    u_g, v_g = Rxy(eta[2]) @ [nu[0], nu[1]]
                    eta[0] += cfg.sim_dynamics.dt * u_g
                    eta[1] += cfg.sim_dynamics.dt * v_g
                    eta[2] = (eta[2] + cfg.sim_dynamics.dt * nu[2]) % (2 * np.pi)
                else:
                    # Maintain forward motion even after path ends
                    warm_start_controls[t] = [base_surge_control, 0.0, 0.0]
            
            # Set warm start control sequence
            controller.u_seq_mean = warm_start_controls
            controller.horizon = horizon_steps
            
            # Step 3: Run diffusion to refine the path
            logger.info('Running diffusion refinement...')
            try:
                goal_y_meters = goal_y / costmap.scale
                # Set target surge speed in controller for forward motion enforcement
                controller.target_surge_speed = cfg.sim_dynamics.target_speed
                controller.enforce_forward_motion = True
                
                controller.DPcontrol(
                    pose=ship_pos_meters,
                    setpoint=goal_pose,
                    dt=cfg.sim_dynamics.dt,
                    nu=nu_start,
                    local_path=local_path,
                    costmap=costmap,
                    goal_y=goal_y_meters
                )
                
                # Get diffusion-refined trajectory
                if hasattr(controller, 'predicted_trajectory'):
                    diffusion_trajectory = controller.predicted_trajectory
                else:
                    logger.error('Controller did not generate trajectory')
                    replan_count += 1
                    continue
                
            except Exception as e:
                logger.error(f'Diffusion controller failed: {e}')
                traceback.print_exc()
                # Fallback to warm start path
                diffusion_trajectory = None
                if path_dir:
                    with open(os.path.join(path_dir, str(replan_count) + '_failed.pkl'), 'wb') as handle:
                        pickle.dump({
                            'replan_count': replan_count,
                            'stamp': t0,
                            'goal': goal_y,
                            'ship_state': ship_pos_scaled,
                            'error': str(e),
                            'raw_message': md.raw_message,
                            'processed_message': md.processed_message
                        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
                replan_count += 1
                continue
            
            logger.info('Diffusion refinement time: {}'.format(time.time() - t))
            logger.info('Total planning time: {}'.format(time.time() - t0))
            
            # Use diffusion trajectory, fallback to warm start if needed
            if diffusion_trajectory is not None and len(diffusion_trajectory) > 0:
                trajectory = diffusion_trajectory
            else:
                logger.warning('Using warm start path as fallback')
                trajectory = warm_start_path

            # Convert trajectory to path format (x, y, psi) in costmap coordinates
            # trajectory is in meters, convert to costmap scale
            path = np.zeros((3, len(trajectory)))
            path[0] = trajectory[:, 0] * costmap.scale  # x
            path[1] = trajectory[:, 1] * costmap.scale  # y
            path[2] = trajectory[:, 2]  # psi (already in radians)

            # Resample path if needed
            if cfg.get('resample_path', False):
                path = resample_path(path, step_size=cfg.get('resample_step_size', 1.0))

            # Send path through pipe (convert to meters and transpose to (N, 3) format)
            path_meters = (path / costmap.scale).T  # Shape: (N, 3)
            md.send_message(path_meters)

            # compute metrics
            compute_time.append(time.time() - t0)
            logger.info('Total compute time: {}'.format(compute_time[-1]))
            logger.info('Average compute time: {}'.format(np.mean(compute_time)))
            
            metrics.put_scalars(
                iteration=replan_count,
                compute_time=compute_time[-1],
                path_length=len(path[0]),
                stamp=t0
            )
            logger.info(metrics.data)
            metrics.step()

            # save path (including warm start and diffusion paths)
            if path_dir:
                warm_start_path_scaled = None
                if warm_start_path is not None:
                    warm_start_path_scaled = np.zeros((3, len(warm_start_path)))
                    warm_start_path_scaled[0] = warm_start_path[:, 0] * costmap.scale
                    warm_start_path_scaled[1] = warm_start_path[:, 1] * costmap.scale
                    warm_start_path_scaled[2] = warm_start_path[:, 2]
                
                with open(os.path.join(path_dir, str(replan_count) + '.pkl'), 'wb') as handle:
                    pickle.dump({
                        'path': path,  # Final diffusion path
                        'trajectory': trajectory,  # In meters
                        'warm_start_path': warm_start_path_scaled,  # Lattice/A* path
                        'replan_count': replan_count,
                        'stamp': t0,
                        'goal': goal_y,
                        'ship_state': ship_pos_scaled,
                        'costmap': costmap.cost_map,
                        'obstacles': costmap.obstacles
                    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

            replan_count += 1

            # plotting
            if cfg.plot.show and replan_count <= cfg.plot.max_iter:
                ship_actual_path[0].append(ship_pos_scaled[0])
                ship_actual_path[1].append(ship_pos_scaled[1])

                if plot is None:
                    plot = plt.figure(figsize=(10, 10))
                    ax = plot.add_subplot(111)

                ax.cla()
                # Plot costmap
                cost_map = costmap.cost_map.copy()
                if cost_map.sum() == 0:
                    cost_map[:] = 1
                ax.imshow(cost_map, origin='lower', cmap='viridis',
                         extent=[0, cfg.map_shape[1] / costmap.scale,
                               0, cfg.map_shape[0] / costmap.scale])
                
                # Plot obstacles
                from matplotlib import patches
                for obs in costmap.obstacles:
                    poly_verts = obs['vertices'] / costmap.scale
                    poly = patches.Polygon(poly_verts, fill=False, edgecolor='cyan', linewidth=0.5)
                    ax.add_patch(poly)
                
                # Plot paths (warm_start_path is in (N, 3) format in meters)
                if warm_start_path is not None and len(warm_start_path) > 0:
                    # warm_start_path is already in meters, just plot it directly
                    ax.plot(warm_start_path[:, 0], warm_start_path[:, 1], 
                           'b--', linewidth=1.5, alpha=0.7, label='Warm Start (A*)')
                
                ax.plot(path[0] / costmap.scale, path[1] / costmap.scale, 'g-', linewidth=2, label='Diffusion Path')
                ax.plot(ship_actual_path[0] / costmap.scale, ship_actual_path[1] / costmap.scale, 
                       'r-', linewidth=1.5, label='Actual Path')
                
                # Plot ship
                R = Rxy(ship_pos_scaled[2])
                ship_poly = ship.vertices @ R.T + [ship_pos_scaled[0] / costmap.scale, 
                                                   ship_pos_scaled[1] / costmap.scale]
                ax.add_patch(patches.Polygon(ship_poly, True, fill=True, color='red'))
                
                ax.set_title(f'Diffusion Planner - Iteration {replan_count}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.legend()
                ax.set_xlim(0, cfg.map_shape[1] / costmap.scale)
                ax.set_ylim(0, cfg.map_shape[0] / costmap.scale)
                
                plt.pause(0.01)

        logger.info('Done diffusion planner!')

    except KeyboardInterrupt:
        logger.info('\033[93mReceived keyboard interrupt!\033[0m')
    except Exception as e:
        logger.error(f'Planner error: {e}')
        traceback.print_exc()
        raise
    finally:
        if md is not None:
            md.close()
        return metrics.get_history() if metrics else []


if __name__ == '__main__':
    # For testing
    pass

