"""Dynamic Positioning demo with diffusion v2 - Uses discrete A* plan as initialization
This version:
1. Gets discrete A* plan (node_path) directly
2. Uses discrete plan as warm start for diffusion
3. Shows discrete plan, baseline, diffusion, and actual trajectories with proper zorder
4. Ensures diffusion trajectory is visible and different from baseline
"""
import pickle
from multiprocessing import Process, Pipe, Queue
from queue import Empty

import matplotlib.pyplot as plt
import numpy as np

from ship_ice_planner import *
from ship_ice_planner.evaluation.evaluate_run_sim import control_vs_time_plot, state_vs_time_plot, tracking_error_plot
from ship_ice_planner.launch import launch
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.controller.model_based_diffusion import ModelBasedDiffusionController
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.primitives import Primitives
from ship_ice_planner.ship import Ship
from ship_ice_planner.swath import generate_swath
from ship_ice_planner.a_star_search import AStar
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict

# Simulation params
SHOW_ANIMATION = True
INF_STREAM = True
SHOW_ICE = True
UPDATE_FREQUENCY_HZ = 2
T = np.inf
FINAL_PLOT_DATA_STEP = 10
DEBUG = False
LOGGING = False

# Full scale parameters
CFG = DotDict.load_from_file(FULL_SCALE_SIM_PARAM_CONFIG)
CFG.output_dir = None

# Diffusion controller configuration
CFG.diffusion = {
    'horizon': 150,
    'num_samples': 128,
    'num_diffusion_steps': 30,  # More steps for better optimization
    'temperature': 0.1,
    'beta0': 1e-4,
    'betaT': 1e-2
}

START = (100, 0, np.pi / 2)
GOAL = [100, 1100]
exp_data = pickle.load(open(FULL_SCALE_SIM_EXP_CONFIG, 'rb'))['exp']
ice_concentration = 0.5
ice_field_idx = 1
obstacles = exp_data[ice_concentration][ice_field_idx]['obstacles']
OBSTACLES = [ob['vertices'] for ob in obstacles]
FLOE_MASSES = [ob['mass'] for ob in obstacles]


def get_discrete_plan_direct(cfg, obstacles, masses, start_pose, goal_y):
    """Get discrete A* plan directly without going through planner process"""
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
        return None, None, None, costmap
    
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
    
    return discrete_plan_meters, continuous_path_meters, costmap, ship


def optimize_with_diffusion_from_discrete(discrete_plan, continuous_path, start_pose, start_nu, dt, target_speed,
                                          costmap, diffusion_cfg, goal_y_full=None):
    """Optimize using diffusion with discrete plan as waypoint anchors"""
    print(f"\n=== Optimizing with diffusion from discrete plan ===")
    print(f"Discrete plan length: {len(discrete_plan)} waypoints")
    print(f"Continuous path length: {len(continuous_path)} waypoints")
    
    # Use continuous path for horizon (not discrete - discrete is too sparse!)
    horizon = diffusion_cfg.get('horizon', 150)
    horizon = min(horizon, len(continuous_path))
    
    # Use continuous path for local_path (better reference)
    if len(continuous_path) > horizon:
        local_path = continuous_path[:horizon].copy()
    else:
        local_path = continuous_path.copy()
        horizon = len(continuous_path)
    
    # Determine goal_y - use full goal if provided, otherwise use local_path endpoint
    if goal_y_full is not None:
        goal_y = goal_y_full
        print(f"Using full goal distance: {goal_y:.1f}m (local_path endpoint: {local_path[-1, 1]:.1f}m)")
    else:
        goal_y = local_path[-1, 1]
    
    # Map discrete plan waypoints to anchor indices in local_path
    discrete_indices = np.linspace(0, horizon - 1, len(discrete_plan)).astype(int)
    waypoint_anchors = {}  # {time_step: (x, y, psi)}
    for i, wp_idx in enumerate(discrete_indices):
        if wp_idx < horizon:
            waypoint_anchors[wp_idx] = discrete_plan[i]
    
    print(f"Using horizon: {horizon} steps (from continuous path)")
    print(f"Mapped {len(waypoint_anchors)} discrete waypoints as anchors")
    
    # Initialize controller
    controller = ModelBasedDiffusionController(
        horizon=horizon,
        num_samples=diffusion_cfg.get('num_samples', 128),
        num_diffusion_steps=diffusion_cfg.get('num_diffusion_steps', 30),
        temperature=diffusion_cfg.get('temperature', 0.1),
        beta0=diffusion_cfg.get('beta0', 1e-4),
        betaT=diffusion_cfg.get('betaT', 1e-2),
        seed=0
    )
    
    controller.target_surge_speed = target_speed
    controller.enforce_forward_motion = True
    
    # Store waypoint anchors for objective function
    controller.waypoint_anchors = waypoint_anchors
    
    # Create warm start from continuous path (better than discrete)
    warm_start_controls = np.zeros((horizon, 3))
    eta = np.array(start_pose, dtype=float)
    current_nu = np.array(start_nu, dtype=float)
    
    from ship_ice_planner.geometry.utils import Rxy
    for t in range(horizon):
        target = local_path[t] if t < len(local_path) else local_path[-1]
        pos_err = target[:2] - eta[:2]
        ang_err = target[2] - eta[2]
        ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
        
        R = Rxy(eta[2])
        body_err = R.T @ pos_err
        
        # Better control initialization
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
    
    print("Running diffusion optimization...")
    # goal_y is already set above (either full goal or local_path endpoint)
    
    # Ensure start_pose and start_nu are numpy arrays
    start_pose_arr = np.array(start_pose, dtype=float)
    start_nu_arr = np.array(start_nu, dtype=float)
    
    # Pass None for goal_y - don't condition on final goal separately
    # The discrete path waypoints already guide to the goal!
    controller.DPcontrol(
        pose=start_pose_arr,
        setpoint=local_path[0] if len(local_path) > 0 else start_pose_arr,
        dt=dt,
        nu=start_nu_arr,
        local_path=local_path,
        costmap=costmap,
        goal_y=None  # Don't condition on goal - discrete path handles it!
    )
    
    # Get optimized trajectory
    if hasattr(controller, 'predicted_trajectory') and controller.predicted_trajectory is not None:
        optimized_traj = controller.predicted_trajectory.copy()
        print(f"✓ Diffusion complete! Trajectory shape: {optimized_traj.shape}")
        print(f"  Trajectory endpoint: {optimized_traj[-1]}")
        
        # CRITICAL: Ensure trajectory actually reaches toward goal
        # If trajectory is too short, extend it to match local_path endpoint
        # Use local_path endpoint as target (discrete path already goes to goal)
        goal_y_target = local_path[-1, 1] if len(local_path) > 0 else goal_y
        final_y = optimized_traj[-1, 1]
        
        # If trajectory doesn't reach close to local_path endpoint (within 50m), extend it
        if abs(final_y - goal_y_target) > 50.0:
            print(f"Warning: Trajectory only reaches y={final_y:.1f}, goal is {goal_y_target:.1f}")
            print(f"Extending trajectory to reach goal...")
            # Extend trajectory linearly toward goal
            num_extra = int(abs(goal_y_target - final_y) / (dt * target_speed)) + 10
            last_point = optimized_traj[-1].copy()
            goal_point = np.array([last_point[0], goal_y_target, last_point[2]])
            extended = np.linspace(last_point, goal_point, num_extra + 1)[1:]
            optimized_traj = np.vstack([optimized_traj, extended])
            print(f"Extended trajectory: {optimized_traj.shape}, endpoint: {optimized_traj[-1]}")
        
        # Extend with remaining continuous path if needed (better than discrete)
        if len(continuous_path) > horizon:
            remaining = continuous_path[horizon:].copy()
            optimized_full = np.vstack([optimized_traj, remaining])
            return optimized_full
        return optimized_traj
    else:
        print("Warning: No trajectory from diffusion, using continuous path")
        return continuous_path[:horizon] if len(continuous_path) >= horizon else continuous_path


def main():
    # Step 1: Get discrete plan directly from A*
    print("=" * 60)
    print("Step 1: Getting discrete A* plan...")
    print("=" * 60)
    
    discrete_plan, continuous_path, costmap, ship = get_discrete_plan_direct(
        CFG, OBSTACLES, FLOE_MASSES, START, GOAL[1]
    )
    
    if discrete_plan is None:
        print("ERROR: Failed to get discrete plan. Using straight line fallback.")
        num_points = 200
        discrete_plan = np.zeros((num_points, 3))
        discrete_plan[:, 0] = START[0]
        discrete_plan[:, 1] = np.linspace(START[1], GOAL[1], num_points)
        discrete_plan[:, 2] = START[2]
        continuous_path = discrete_plan.copy()
    
    print(f"Discrete plan: {discrete_plan.shape}")
    print(f"Continuous path: {continuous_path.shape}")
    
    # Step 2: Optimize with diffusion using discrete plan as initialization
    print("\n" + "=" * 60)
    print("Step 2: Optimizing with diffusion (discrete plan initialization)...")
    print("=" * 60)
    
    optimized_path = optimize_with_diffusion_from_discrete(
        discrete_plan=discrete_plan,
        continuous_path=continuous_path,
        start_pose=START,
        start_nu=[0, 0, 0],
        dt=CFG.sim_dynamics.dt,
        target_speed=CFG.sim_dynamics.target_speed,
        costmap=costmap,
        diffusion_cfg=CFG.diffusion,
        goal_y_full=GOAL[1]  # Pass full goal distance so diffusion optimizes for full distance
    )
    
    print(f"\nOptimized path: {optimized_path.shape}")
    print(f"  Start: {optimized_path[0]}")
    print(f"  End: {optimized_path[-1]}")
    
    # Step 3: Track optimized path with standard controller
    print("\n" + "=" * 60)
    print("Step 3: Tracking optimized path...")
    print("=" * 60)
    
    sim_dynamics = SimShipDynamics(
        eta=START, nu=[0, 0, 0],
        output_dir=CFG.output_dir,
        **CFG.sim_dynamics
    )
    
    sim_dynamics.init_trajectory_tracking(optimized_path)
    state = sim_dynamics.state
    
    # Setup plotting
    plot = None
    running = True
    if SHOW_ANIMATION:
        if SHOW_ICE:
            show_obs = OBSTACLES
        else:
            show_obs = []
        
        # Plot WITHOUT legend initially (we'll add custom one)
        plot = Plot(obstacles=show_obs, path=optimized_path.T, legend=False, track_fps=True, 
                    y_axis_limit=CFG.plot.y_axis_limit,
                    ship_vertices=CFG.ship.vertices, target=sim_dynamics.setpoint[:2], 
                    inf_stream=INF_STREAM,
                    ship_pos=state.eta, map_figsize=None, sim_figsize=(10, 10), 
                    remove_sim_ticks=False, goal=GOAL[1],
                    map_shape=CFG.map_shape)

        def on_close(event):
            nonlocal running
            if event.key == 'escape':
                running = False

        plot.sim_fig.canvas.mpl_connect('key_press_event', on_close)
        
        # Plot all paths with high zorder AFTER obstacles are initialized
        # Note: Plot class adds obstacles as PatchCollection (usually zorder=1)
        # We need to plot our comparison paths AFTER Plot initialization with HIGH zorder
        if hasattr(plot, 'sim_ax'):
            # Update existing path line from Plot to be our optimized path
            if hasattr(plot, 'path_line'):
                plot.path_line.set_data(optimized_path[:, 0], optimized_path[:, 1])
                plot.path_line.set_label('Diffusion Optimized')
                plot.path_line.set_color('green')
                plot.path_line.set_linewidth(4)
                plot.path_line.set_zorder(10)
            
            # Add comparison paths with HIGH zorder (after obstacles, before ship)
            # Discrete plan (magenta, markers) - high zorder
            discrete_line, = plot.sim_ax.plot(discrete_plan[:, 0], discrete_plan[:, 1], 
                           'mo-', linewidth=2.5, markersize=5, 
                           label='Discrete A* Plan', alpha=0.85, zorder=9,
                           markevery=max(1, len(discrete_plan)//30))
            # Continuous baseline (blue, dashed) - high zorder  
            baseline_line, = plot.sim_ax.plot(continuous_path[:, 0], continuous_path[:, 1], 
                           'b--', linewidth=2.5, label='Continuous Baseline', 
                           alpha=0.75, zorder=8)
            
            # Ensure ship patch is on top
            if hasattr(plot, 'ship_patch'):
                plot.ship_patch.set_zorder(15)
            
            # Create single unified legend
            # Remove default legend if Plot created one, create our own
            if hasattr(plot.sim_ax, 'legend_') and plot.sim_ax.legend_ is not None:
                plot.sim_ax.legend_.remove()
            legend = plot.sim_ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            # Set zorder on the legend object (not as parameter)
            legend.set_zorder(20)

    ship_actual_path = ([], [])
    steps = 0
    goal_reached = False

    try:
        while T >= steps and not goal_reached and running:
            steps += 1
            
            if state.y >= GOAL[1]:
                print(f"\n✓ Goal reached at step {steps}! y={state.y:.2f} >= goal_y={GOAL[1]}")
                goal_reached = True
                break
            
            sim_dynamics.control()
            sim_dynamics.log_step()
            sim_dynamics.sim_step()
            state.integrate()

            if SHOW_ANIMATION and steps % (1 / CFG.sim_dynamics.dt / UPDATE_FREQUENCY_HZ) == 0:
                ship_actual_path[0].append(state.x)
                ship_actual_path[1].append(state.y)

                plot.update_path(optimized_path.T, ship_state=ship_actual_path, target=sim_dynamics.setpoint[:2])
                plot.update_ship(CFG.ship.vertices, state.x, state.y, state.psi)

                fps = plot.update_fps()

                plot.title_text.set_text(
                    f'FPS: {fps:.0f}, '
                    f'Real time speed: {fps / UPDATE_FREQUENCY_HZ:.1f}x, '
                    f'Time: {sim_dynamics.sim_time:.1f} s\n'
                    f'Tracking: Diffusion Optimized Path\n'
                    f'surge {state.u:.2f} (m/s), '
                    f'sway {-state.v:.2f} (m/s), '
                    f'yaw rate {-state.r * 180 / np.pi:.2f} (deg/s)'
                )

                plot.animate_sim()

    except KeyboardInterrupt:
        print('Received keyboard interrupt, exiting...')

    finally:
        print(f'\nDone! Total steps: {steps}')

        if plot is not None:
            plot.close()

        sim_data = sim_dynamics.get_state_history().iloc[::FINAL_PLOT_DATA_STEP]

        # Generate comparison plots
        print("\nGenerating comparison plots...")
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10))
        
        # Plot obstacles
        from matplotlib import patches
        for obs in OBSTACLES:
            poly = patches.Polygon(obs, fill=True, edgecolor='gray', facecolor='lightblue', 
                                  linewidth=1, alpha=0.5, zorder=1)
            ax1.add_patch(poly)
        
        # Plot all paths with proper zorder
        ax1.plot(discrete_plan[:, 0], discrete_plan[:, 1], 
               'mo-', linewidth=3, markersize=8, label='Discrete A* Plan', 
               alpha=0.9, zorder=6, markevery=max(1, len(discrete_plan)//30))
        ax1.plot(continuous_path[:, 0], continuous_path[:, 1], 
               'b--', linewidth=3, label='Continuous Baseline', alpha=0.8, zorder=5)
        ax1.plot(optimized_path[:, 0], optimized_path[:, 1], 
               'g-', linewidth=4, label='Diffusion Optimized', alpha=0.95, zorder=7)
        
        # Actual executed path
        actual_path_data = sim_data[['x', 'y']].values
        ax1.plot(actual_path_data[:, 0], actual_path_data[:, 1], 
               'r-', linewidth=3, label='Actual Executed', alpha=0.9, zorder=8)
        
        ax1.scatter([START[0]], [START[1]], c='green', s=150, marker='o', 
                   label='Start', zorder=10, edgecolors='black', linewidths=2)
        ax1.scatter([GOAL[0]], [GOAL[1]], c='red', s=200, marker='*', 
                   label='Goal', zorder=10, edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title('Trajectory Comparison: Discrete A* | Continuous | Diffusion | Actual', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('diffusion_dynamic_comparison_v2.png', dpi=150, bbox_inches='tight')
        print("Saved comparison plot to: diffusion_dynamic_comparison_v2.png")
        plt.show()
        
        # Standard plots
        state_vs_time_plot(sim_data)
        control_vs_time_plot(sim_data,
                             dim_U=len(sim_dynamics.state.u_control),
                             control_labels=sim_dynamics.vessel_model.controls)
        
        if CFG.get('max_replan') == 1:
            tracking_error_plot(sim_data, optimized_path, CFG.map_shape)
        
        plt.show()

        print('Done')


if __name__ == '__main__':
    print('=' * 60)
    print('Dynamic Positioning Demo with Diffusion v2')
    print('=' * 60)
    print('1. Get discrete A* plan directly')
    print('2. Use discrete plan as initialization for diffusion')
    print('3. Track diffusion-optimized path')
    print('=' * 60)
    main()

