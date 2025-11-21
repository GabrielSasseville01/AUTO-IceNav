""" Main script for running simulation experiments with autonomous ship navigation in ice """
import math
import os
import random
import time
from multiprocessing import Process, Pipe, Queue
from queue import Empty, Full

import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import pyplot as plt
from pymunk import Vec2d
import pymunk.batch

from ship_ice_planner.evaluation.evaluate_run_sim import floe_mass_hist_plot, ke_impulse_vs_time_plot, \
    control_vs_time_plot, state_vs_time_plot, impact_locs_impulse_plot, fracturing_stats_plot, ridging_stats_plot
from ship_ice_planner.launch import launch
from ship_ice_planner.controller.sim_dynamics import SimShipDynamics
from ship_ice_planner.geometry.polygon import poly_area
from ship_ice_planner.geometry.utils import get_global_obs_coords
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict
from ship_ice_planner.utils.sim_utils import *
from ship_ice_planner.utils.ice_fracture_ridge import (
    IceFloeState, RidgeZone, FRACTURE_ENABLED, RIDGING_ENABLED,
    fracture_ice_floe, check_ridging_conditions, create_ridge_zone,
    compute_ridge_resistance, compute_compression_force,
    RIDGE_MIN_FLOES, RIDGE_ACCUMULATION_RATE,
    RIDGE_DECAY_TIME, RIDGE_MIN_ACTIVITY_DISTANCE
)


# global vars for dir/file names
PLOT_DIR = 'sim_plots'
TRIAL_SIM_PLOT = 'sim.pdf'
FLOE_MASS_HIST_PLOT = 'floe_mass_hist.pdf'
KE_IMPULSE_VS_TIME_PLOT = 'ke_impulse_vs_time.pdf'
IMPACT_LOCS_IMPULSE_PLOT = 'impact_locations_impulse.pdf'
STATE_VS_TIME_PLOT = 'state_vs_time.pdf'
CONTROL_VS_TIME_PLOT = 'control_vs_time.pdf'
FRACTURING_PLOT = 'fracturing_stats.pdf'
RIDGING_PLOT = 'ridging_stats.pdf'


def sim(
        cfg_file: str = None,
        cfg: DotDict = None,
        debug: bool = False,
        logging: bool = False,
        log_level: int = 10,
        init_queue: dict = None
):
    if cfg_file:
        # load config
        cfg = DotDict.load_from_file(cfg_file)
        cfg.cfg_file = cfg_file

    if cfg.output_dir:
        os.makedirs(cfg.output_dir)

    # multiprocessing setup
    queue = Queue(maxsize=1)  # LIFO queue to send state information to planner
    conn_recv, conn_send = Pipe(duplex=False)  # pipe to send new path to controller and for plotting
    planner = Process(target=launch,
                      kwargs=dict(cfg=cfg,
                                  debug=debug,
                                  logging=logging,
                                  log_level=log_level,
                                  pipe=conn_send,
                                  queue=queue))
    planner.start()

    # get sim params from config
    t_max = cfg.sim.t_max if cfg.sim.t_max else np.inf  # max number of main loop iterations
    dt = cfg.sim_dynamics.dt
    steps = cfg.sim.steps

    # seed to make sim deterministic
    seed = cfg.get('seed', 0)
    np.random.seed(seed)
    random.seed(seed)

    # timeout for polling of path updates
    # if timeout > 0, then the simulation loop will block until
    # a new path is received after a replan has been triggered
    planner_timeout = cfg.sim.get('planner_timeout', 0.0)

    # get goal, start, and obstacles
    if init_queue:
        goal = init_queue['goal']
        start = init_queue['ship_state']
        obs_dicts = init_queue['obstacles']

    else:
        goal, start = cfg.ship.goal_pos, cfg.ship.start_pos
        # generate obstacles and an associated polygon object
        obs_dicts, _ = generate_obstacles(**cfg.obstacle, seed=seed)

    obstacles = [ob['vertices'] for ob in obs_dicts]
    sqrt_floe_areas = np.array([math.sqrt(poly_area(ob['vertices'])) for ob in obs_dicts])  # for drag calculation

    # setup controller and simulated ship dynamics
    sim_dynamics = SimShipDynamics(
        eta=start, nu=[0, 0, 0],
        output_dir=cfg.output_dir,
        **cfg.sim_dynamics
    )
    state = sim_dynamics.state

    # setup pymunk environment
    space = init_pymunk_space()

    # setup boundaries of ice field
    # create_channel_borders(space, *cfg.map_shape)  # disabling for now, causes weird behavior

    # initialize collision metrics
    system_ke_loss = []   # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_ke
                          # source code in Chimpunk2D cpArbiterTotalKE
    delta_ke_ice = []
    total_impulse = []    # https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.total_impulse
                          # source code in Chimpunk2D cpArbiterTotalImpulse
    collided_ob_idx = []  # idx for the collided obstacle, assumes id corresponds to the index in obs_dicts
    contact_pts = []      # collision contact points in the local coordinates of the ship

    # setup pymunk collision callbacks
    def pre_solve_handler(arbiter, space, data):
        ice_body = arbiter.shapes[1].body
        ice_body.pre_collision_KE = ice_body.kinetic_energy  # hacky, adding a field to pymunk body object
        return True

    def post_solve_handler(arbiter, space, data):
        nonlocal system_ke_loss, delta_ke_ice, total_impulse, collided_ob_idx, contact_pts
        nonlocal polygons, poly_vertices, ice_floe_states, floe_masses
        ship_shape, ice_shape = arbiter.shapes

        # update metrics
        system_ke_loss.append(arbiter.total_ke)
        delta_ke_ice.append(ice_shape.body.kinetic_energy - ice_shape.body.pre_collision_KE)
        total_impulse.append(arbiter.total_impulse)
        collided_ob_idx.append(ice_shape.idx)

        # find the impact locations in the local coordinates of the ship
        # see https://stackoverflow.com/a/78017626 and run demo below to understand contact points
        # https://github.com/viblo/pymunk/blob/master/pymunk/examples/collisions.py
        if len(arbiter.contact_point_set.points) == 2:  # max 2 contact points
            # take the average
            c1, c2 = arbiter.contact_point_set.points
            contact_pts.append(list(ship_shape.body.world_to_local((c1.point_b + c2.point_b) / 2)))
        else:
            c1 = arbiter.contact_point_set.points[0]
            contact_pts.append(list(ship_shape.body.world_to_local(c1.point_b)))
        
        # Track impulse for fracturing
        if FRACTURE_ENABLED and ice_shape.idx in ice_floe_states:
            impulse_magnitude = np.linalg.norm(arbiter.total_impulse)
            ice_state = ice_floe_states[ice_shape.idx]
            ice_state.add_impulse(impulse_magnitude)
            
            # Check if floe should fracture (defer to end of step to avoid modifying during collision)
            # We'll check this after the physics step

    handler = space.add_collision_handler(1, 2)
    handler.pre_solve = pre_solve_handler
    handler.post_solve = post_solve_handler

    # init pymunk physics objects
    ship_shape = create_sim_ship(space,  # polygon for ship
                                 cfg.ship.vertices,
                                 state.eta,
                                 body_type=pymunk.Body.KINEMATIC,
                                 velocity=state.get_global_velocity())
    ship_shape.collision_type = 1  # for collision callbacks
    ship_body = ship_shape.body

    # polygons for ice floes
    polygons = generate_sim_obs(space, obs_dicts)
    poly_vertices = []  # generate these once
    ice_floe_states = {}  # Track floe states for fracturing
    next_floe_idx = len(polygons)  # Track next available index for new floes
    for idx, p in enumerate(polygons):
        p.collision_type = 2  # for collision callbacks
        vertices = list_vec2d_to_numpy(p.get_vertices())
        poly_vertices.append(
            np.concatenate([vertices, [vertices[0]]])  # close the polygon (speeds up plotting somehow...)
        )
        p.idx = idx  # hacky but makes it easier to keep track which polygon collided
        
        # Initialize ice floe state for fracturing
        if FRACTURE_ENABLED:
            floe_area = poly_area(vertices)
            ice_floe_states[p.idx] = IceFloeState(p, floe_area)
    
    floe_masses = np.array([poly.mass for poly in polygons])  # may differ slightly from the mass in the exp config
    
    # Ridge tracking
    ridge_zones = []  # List of active ridge zones
    ridge_check_counter = 0  # Counter for periodic ridge checking
    
    # Tracking for plots
    fracture_history = []  # List of fracture events: {'time', 'x', 'y', 'impulse', 'num_pieces', 'num_floes'}
    ridge_history = []  # List of ridge creation events: {'time', 'x', 'y', 'direction'}
    ridge_thickness_history = []  # List of thickness updates: {'time', 'thickness' (list), 'num_ridges'}
    ridge_resistance_history = []  # List of resistance values: {'time', 'resistance'}
    
    # Visualization sync tracking
    vis_sync_counter = 0  # Counter for periodic visualization re-sync
    ridge_geometry_update_counter = 0  # Counter for dynamic geometry updates

    # uses new batch module https://www.pymunk.org/en/latest/pymunk.batch.html
    buffer_get_body = pymunk.batch.Buffer()
    batch_fields_get_body = (pymunk.batch.BodyFields.POSITION
                             | pymunk.batch.BodyFields.ANGLE
                             | pymunk.batch.BodyFields.VELOCITY
                             | pymunk.batch.BodyFields.ANGULAR_VELOCITY
                             )
    batch_dim_get_body = 6

    buffer_set_body = pymunk.batch.Buffer()
    batch_fields_set_body = (pymunk.batch.BodyFields.VELOCITY
                             | pymunk.batch.BodyFields.ANGULAR_VELOCITY
                             )

    # send first message
    queue.put(dict(
        goal=[goal[0],
              goal[1] + cfg.get('goal_offset', 0)],  # option to offset goal
        ship_state=start,
        obstacles=obstacles,
        masses=floe_masses,
        # metadata=dict(  # can optionally send metadata
        # )
    ))

    # get path
    path = conn_recv.recv()
    path = np.asarray(path)

    # setup trajectory tracking
    sim_dynamics.init_trajectory_tracking(path)

    # initialize plotting / animation
    plot = None
    running = True
    save_fig_dir = os.path.join(cfg.output_dir, PLOT_DIR) if cfg.output_dir else None
    if save_fig_dir:
        os.makedirs(save_fig_dir, exist_ok=True)
        print(f'Plots will be saved to: {save_fig_dir}')
    if cfg.anim.show or cfg.anim.save:
        plot = Plot(obstacles=obstacles, path=path.T, legend=False, track_fps=True, y_axis_limit=cfg.plot.y_axis_limit,
                    ship_vertices=cfg.ship.vertices, target=sim_dynamics.setpoint[:2], inf_stream=cfg.anim.inf_stream,
                    ship_pos=state.eta, map_figsize=None, sim_figsize=(5, 5), remove_sim_ticks=True, goal=goal[1],
                    save_fig_dir=save_fig_dir, map_shape=cfg.map_shape,
                    save_animation=cfg.anim.save, anim_fps=cfg.anim.fps
                    )
        def on_close(event):
            nonlocal running
            if event.key == 'escape':
                running = False

        plot.sim_fig.canvas.mpl_connect('key_press_event', on_close)

    # for plotting ship path
    ship_actual_path = ([], [])  # list for x and y

    iteration = 0    # to keep track of simulation steps
    t = time.time()  # to keep track of time taken for each sim iteration

    try:
        # main simulation loop
        while True:
            # check termination conditions
            if iteration >= t_max:
                print('\nReached max number of iterations:', t_max)
                break

            if running is False:
                print('\nReceived stop signal, exiting...')
                break

            if state.y >= goal[1]:
                print('\nReached goal:', goal)
                break

            time_per_step = time.time() - t
            t = time.time()

            # get batched body data
            buffer_get_body.clear()
            pymunk.batch.get_space_bodies(space, batch_fields_get_body, buffer_get_body)
            batched_data = np.asarray(
                list(memoryview(buffer_get_body.float_buf()).cast('d'))[batch_dim_get_body:]  # ignore ship body
            ).reshape(-1, batch_dim_get_body)

            # get updated ship pose
            state.eta = (ship_body.position.x, ship_body.position.y, ship_body.angle)

            if iteration % int(1 / dt) == 0:
                print(
                    f'Simulation step {iteration} / {t_max}, '
                    f'ship eta ({state.x:.2f}m, {state.y:.2f}m, {state.psi:.2f}rad), '
                    f'sim time {sim_dynamics.sim_time:.2f} s, '
                    f'time per step {time_per_step:.4f} s ',
                    end='\r',
                )

            if planner.is_alive():
                if sim_dynamics.check_trigger_replan():
                    # empty queue to ensure latest state data is pushed
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                    try:
                        # Ensure poly_vertices matches batched_data order for planner
                        # This is critical - wrong order means planner sees obstacles in wrong positions
                        ice_polygons_for_planner = [p for p in polygons if p.collision_type == 2]
                        if len(ice_polygons_for_planner) == len(batched_data) and len(batched_data) > 0:
                            # Match polygons to batched_data by position
                            planner_vertices = []
                            used_indices = set()
                            
                            for i in range(len(batched_data)):
                                body_pos = batched_data[i, :2]
                                best_idx = None
                                best_dist = 30.0
                                
                                for j, p in enumerate(ice_polygons_for_planner):
                                    if j in used_indices:
                                        continue
                                    poly_pos = np.array([p.body.position.x, p.body.position.y])
                                    dist = np.linalg.norm(body_pos - poly_pos)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = j
                                
                                if best_idx is not None:
                                    p = ice_polygons_for_planner[best_idx]
                                    used_indices.add(best_idx)
                                    vertices = list_vec2d_to_numpy(p.get_vertices())
                                    planner_vertices.append(
                                        np.concatenate([vertices, [vertices[0]]])
                                    )
                                elif i < len(poly_vertices):
                                    planner_vertices.append(poly_vertices[i])
                            
                            # Use matched vertices if we got all matches
                            if len(planner_vertices) == len(batched_data):
                                planner_poly_vertices = planner_vertices
                                planner_masses = np.array([ice_polygons_for_planner[j].mass for j in sorted(used_indices)][:len(batched_data)]) if len(used_indices) == len(batched_data) else floe_masses[:len(batched_data)]
                            else:
                                planner_poly_vertices = poly_vertices
                                planner_masses = floe_masses
                        else:
                            planner_poly_vertices = poly_vertices
                            planner_masses = floe_masses
                        
                        queue.put(dict(
                            ship_state=state.eta,
                            obstacles=(planner_poly_vertices,  # matched to batched_data order
                                       batched_data[:, :2],  # updated positions
                                       batched_data[:, 2]),  # updated angles
                            masses=planner_masses
                        ), timeout=1)

                        if planner_timeout:
                            conn_recv.poll(planner_timeout)  # wait for new path, block until timeout

                    except Full:
                        # we should only be reaching this point in rare cases!
                        print('Something went wrong with sending message to planner...')
                        pass

                # check for path
                if conn_recv.poll():
                    msg = conn_recv.recv()
                    # confirm we have path
                    if len(msg) == 0:
                        print('Error, received empty path!')
                    else:
                        new_path = np.asarray(msg)  # n x 3
                        # confirm path is a minimum of 2 points
                        if len(new_path) > 1:
                            path = new_path
                            sim_dynamics.setpoint_generator.replan_update(
                                state.get_vessel_speed(), (state.x, state.y), path
                            )

            # update controller
            sim_dynamics.control()

            # compute collision forces on ship
            tau_env = compute_collision_force_on_ship(total_impulse,
                                                      contact_pts,
                                                      state.u, state.v, state.psi,
                                                      dt / steps)
            
            # Add ridge resistance if ridging is enabled
            if RIDGING_ENABLED and len(ridge_zones) > 0:
                ship_velocity_vec = Vec2d(*state.get_global_velocity())
                ship_position_vec = Vec2d(state.x, state.y)
                ridge_resistance = compute_ridge_resistance(
                    ship_velocity_vec, ship_position_vec, 
                    cfg.ship.vertices, state.psi,  # Ship geometry for contact area
                    ridge_zones, dt, current_time=t
                )
                
                # Track resistance
                resistance_magnitude = np.linalg.norm([ridge_resistance[0], ridge_resistance[1]])
                ridge_resistance_history.append({
                    'time': t,
                    'resistance': resistance_magnitude
                })
                
                # Convert ridge resistance to ship body frame
                from ship_ice_planner.geometry.utils import Rxy
                R = Rxy(state.psi).T
                ridge_resistance_body = R @ np.array([ridge_resistance[0], ridge_resistance[1]])
                tau_env[0] += ridge_resistance_body[0]  # Add to surge
                tau_env[1] += ridge_resistance_body[1]  # Add to sway
                tau_env[2] += ridge_resistance[2]  # Add to yaw moment
            
            sim_dynamics.vessel_model.set_environment_force(tau_env)

            # store simulation data
            sim_dynamics.log_step(
                system_ke_loss=system_ke_loss,
                delta_ke_ice=delta_ke_ice,
                total_impulse=total_impulse,
                collided_ob_idx=collided_ob_idx,
                contact_pts=contact_pts,
                tau_env=tau_env,
                real_time=t
            )

            if iteration % cfg.anim.plot_steps == 0 and plot is not None:
                try:
                    # rendering is slow!! better option would be to use a proper library...
                    ship_actual_path[0].append(state.x)
                    ship_actual_path[1].append(state.y)

                    plot.update_path(path.T, target=sim_dynamics.setpoint[:2])  # , ship_state=ship_actual_path)
                    plot.update_ship(cfg.ship.vertices, *state.eta,
                                     move_yaxis_threshold=cfg.anim.move_yaxis_threshold)
                    
                    # Update obstacles for visualization
                    plot.update_obstacles(obstacles=get_global_obs_coords(  # updating obs is very slow...
                        poly_vertices,
                        batched_data[:, :2],  # updated positions
                        batched_data[:, 2])   # updated angles
                    )
                    

                    fps = plot.update_fps()

                    plot.title_text.set_text(
                        f'FPS: {fps:.0f}, '
                        f'Real time speed: {fps * dt * cfg.anim.plot_steps:.1f}x, '
                        f'Time: {sim_dynamics.sim_time:.1f} s\n'
                        f'surge {state.u:.2f} (m/s), '
                        f'sway {-state.v:.2f} (m/s), '                    # in body frame, positive sway is to the right
                        f'yaw rate {-state.r * 180 / np.pi:.2f} (deg/s)'  # in body frame, positive yaw is clockwise
                        # in the sim, these are reversed! so for plotting purposes flip the sign
                    )

                    plot.animate_sim()
                except Exception as e:
                    # Don't let visualization errors stop the simulation
                    if debug:
                        print(f"Visualization error (non-fatal): {e}")
                    pass

            # simulate ship dynamics
            sim_dynamics.sim_step()

            # apply velocity commands to ship body
            ship_body.velocity = Vec2d(*state.get_global_velocity())
            ship_body.angular_velocity = state.r

            # Check for fracturing BEFORE applying drag (to avoid shape mismatches)
            if FRACTURE_ENABLED:
                floes_to_fracture = []
                for idx, ice_state in list(ice_floe_states.items()):
                    if ice_state.should_fracture():
                        # Find the shape
                        ice_shape = None
                        for poly in polygons:
                            if hasattr(poly, 'idx') and poly.idx == idx:
                                ice_shape = poly
                                break
                        if ice_shape:
                            floes_to_fracture.append((ice_shape, ice_state))
                
                # Fracture floes
                for ice_shape, ice_state in floes_to_fracture:
                    # Record fracture event before fracturing
                    fracture_pos = ice_shape.body.position
                    fracture_impulse = ice_state.cumulative_impulse
                    
                    new_floes, next_floe_idx = fracture_ice_floe(space, ice_shape, ice_state, ice_floe_states, next_floe_idx)
                    
                    if new_floes:
                        # Track fracture event
                        # Count current floes (before adding new ones)
                        current_floe_count = len([p for p in polygons if p.collision_type == 2])
                        fracture_history.append({
                            'time': t,
                            'x': fracture_pos.x,
                            'y': fracture_pos.y,
                            'impulse': fracture_impulse,
                            'num_pieces': len(new_floes),
                            'num_floes': current_floe_count + len(new_floes) - 1  # Current count + new - old
                        })
                        # Update polygon lists
                        if hasattr(ice_shape, 'idx'):
                            polygons = [p for p in polygons if not hasattr(p, 'idx') or p.idx != ice_shape.idx]  # Remove old
                        polygons.extend(new_floes)  # Add new
                        
                        # Re-read batched data after fracturing to match new polygon count
                        buffer_get_body.clear()
                        pymunk.batch.get_space_bodies(space, batch_fields_get_body, buffer_get_body)
                        batched_data = np.asarray(
                            list(memoryview(buffer_get_body.float_buf()).cast('d'))[batch_dim_get_body:]
                        ).reshape(-1, batch_dim_get_body)
                        
                        # Update poly_vertices, floe_masses, and sqrt_floe_areas
                        # Match polygons to batched_data order by position to prevent visualization glitches
                        ice_polygons_only = [p for p in polygons if p.collision_type == 2]
                        if len(ice_polygons_only) == len(batched_data):
                            # Match each batched_data body to closest polygon
                            poly_vertices = []
                            used_indices = set()
                            
                            for i in range(len(batched_data)):
                                body_pos = batched_data[i, :2]
                                best_idx = None
                                best_dist = 20.0  # 20m max distance
                                
                                for j, p in enumerate(ice_polygons_only):
                                    if j in used_indices:
                                        continue
                                    poly_pos = np.array([p.body.position.x, p.body.position.y])
                                    dist = np.linalg.norm(body_pos - poly_pos)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = j
                                
                                if best_idx is not None:
                                    p = ice_polygons_only[best_idx]
                                    used_indices.add(best_idx)
                                    vertices = list_vec2d_to_numpy(p.get_vertices())
                                    poly_vertices.append(
                                        np.concatenate([vertices, [vertices[0]]])
                                    )
                                else:
                                    # If no match, just use first available (shouldn't happen)
                                    if len(ice_polygons_only) > i:
                                        p = ice_polygons_only[i]
                                        vertices = list_vec2d_to_numpy(p.get_vertices())
                                        poly_vertices.append(
                                            np.concatenate([vertices, [vertices[0]]])
                                        )
                            
                            # Update masses and areas in same order as poly_vertices
                            # Build matched_polygons list by finding which polygon was used for each poly_vertex
                            matched_polygons = []
                            for i in range(len(poly_vertices)):
                                # Find polygon that matches this vertex set by checking position
                                body_pos = batched_data[i, :2] if i < len(batched_data) else None
                                best_match = None
                                if body_pos is not None:
                                    for p in ice_polygons_only:
                                        poly_pos = np.array([p.body.position.x, p.body.position.y])
                                        if np.linalg.norm(poly_pos - body_pos) < 2.0:  # 2m tolerance
                                            best_match = p
                                            break
                                # Use best match or fallback
                                if best_match is not None:
                                    matched_polygons.append(best_match)
                                elif i < len(ice_polygons_only):
                                    matched_polygons.append(ice_polygons_only[i])
                            
                            # Ensure we have enough
                            while len(matched_polygons) < len(batched_data) and len(matched_polygons) < len(ice_polygons_only):
                                matched_polygons.append(ice_polygons_only[len(matched_polygons)])
                            
                            floe_masses = np.array([poly.mass for poly in matched_polygons[:len(batched_data)]])
                            sqrt_floe_areas = np.array([math.sqrt(poly_area(list_vec2d_to_numpy(p.get_vertices()))) for p in matched_polygons[:len(batched_data)]])
                        else:
                            # Fallback: just rebuild in polygon order
                            poly_vertices = []
                            for p in ice_polygons_only:
                                vertices = list_vec2d_to_numpy(p.get_vertices())
                                poly_vertices.append(
                                    np.concatenate([vertices, [vertices[0]]])
                                )
                            floe_masses = np.array([poly.mass for poly in ice_polygons_only])
                            sqrt_floe_areas = np.array([math.sqrt(poly_area(list_vec2d_to_numpy(p.get_vertices()))) for p in ice_polygons_only])

            # apply forces on ice floes
            # Ensure shapes match before applying drag
            # Re-read batched data to ensure it matches current polygon count
            buffer_get_body.clear()
            pymunk.batch.get_space_bodies(space, batch_fields_get_body, buffer_get_body)
            batched_data = np.asarray(
                list(memoryview(buffer_get_body.float_buf()).cast('d'))[batch_dim_get_body:]
            ).reshape(-1, batch_dim_get_body)
            
            # Ensure sqrt_floe_areas and floe_masses match batched_data length
            # Note: batched_data includes all dynamic bodies (ice floes), but may exclude static bodies (ridge zones)
            num_bodies = len(batched_data)
            num_ice_polygons = len([p for p in polygons if p.collision_type == 2])  # Only count ice floes
            
            if len(sqrt_floe_areas) != num_bodies or len(floe_masses) != num_bodies:
                # Recalculate to match current number of bodies
                # Filter to only ice floe polygons (collision_type == 2)
                ice_polygons = [p for p in polygons if p.collision_type == 2]
                if len(ice_polygons) >= num_bodies:
                    sqrt_floe_areas = np.array([math.sqrt(poly_area(list_vec2d_to_numpy(p.get_vertices()))) for p in ice_polygons[:num_bodies]])
                    floe_masses = np.array([poly.mass for poly in ice_polygons[:num_bodies]])
                elif len(ice_polygons) < num_bodies:
                    # If we have fewer polygons than bodies, pad with dummy values
                    # This can happen if bodies were added outside our tracking
                    sqrt_floe_areas_padded = np.array([math.sqrt(poly_area(list_vec2d_to_numpy(p.get_vertices()))) for p in ice_polygons])
                    floe_masses_padded = np.array([poly.mass for poly in ice_polygons])
                    # Pad to match num_bodies
                    if len(sqrt_floe_areas_padded) < num_bodies:
                        pad_size = num_bodies - len(sqrt_floe_areas_padded)
                        sqrt_floe_areas = np.concatenate([sqrt_floe_areas_padded, np.ones(pad_size) * 1.0])
                        floe_masses = np.concatenate([floe_masses_padded, np.ones(pad_size) * 1000.0])
                    else:
                        sqrt_floe_areas = sqrt_floe_areas_padded[:num_bodies]
                        floe_masses = floe_masses_padded[:num_bodies]
                else:
                    # Fallback
                    sqrt_floe_areas = np.ones(num_bodies) * 1.0
                    floe_masses = np.ones(num_bodies) * 1000.0
            
            if len(sqrt_floe_areas) == len(batched_data) and len(floe_masses) == len(batched_data):
                batched_apply_drag_from_water(sqrt_floe_areas,
                                              floe_masses,
                                              batched_data[:, 3:5],  # updated velocities
                                              batched_data[:, 5],    # updated angular velocities
                                              [*ship_body.velocity, ship_body.angular_velocity],
                                              dt,
                                              buffer_set_body,
                                              )
                pymunk.batch.set_space_bodies(
                    space, batch_fields_set_body, buffer_set_body
                )
            # apply_random_disturbance(polygons)  # these are not optimized using batch module
            # apply_current(polygons, cfg.sim.ocean_current)  # (not fully tested! would need to also update sim_dynamics.py)
            
            # Check for ridging conditions (periodically to avoid performance hit)
            if RIDGING_ENABLED:
                try:
                    ridge_check_counter += 1
                    if ridge_check_counter >= 10:  # Check every 10 iterations
                        ridge_check_counter = 0
                        ship_velocity_vec = Vec2d(*state.get_global_velocity())
                        ridging_candidates, compressed_groups = check_ridging_conditions(
                            polygons, ship_body.position, ship_velocity_vec, state.psi
                        )
                    
                    # NEW: Use floe-to-floe compression detection (more realistic)
                    # Create ridge zone for each compressed group with enough floes
                    # Allow groups of 2+ floes (compression requires at least 2)
                    for compressed_group in compressed_groups:
                        if len(compressed_group) >= 2:  # Minimum 2 floes for compression
                            # Calculate center of compressed floe group
                            group_positions = [poly.body.position for poly in compressed_group]
                            ridge_center = Vec2d(
                                sum(p.x for p in group_positions) / len(group_positions),
                                sum(p.y for p in group_positions) / len(group_positions)
                            )
                            
                            # Calculate ridge orientation from compression direction
                            # Use average direction from ship to floes
                            ship_to_center_vec = ridge_center - ship_body.position
                            if ship_to_center_vec.length > 0.01:  # Avoid zero-length vector
                                ship_to_center = ship_to_center_vec.normalized()
                                ridge_direction = math.atan2(ship_to_center.y, ship_to_center.x)
                            else:
                                # Use ship heading if center is too close
                                ridge_direction = state.psi
                            
                            # Estimate ridge size from floe group extent
                            positions_array = np.array([[p.x, p.y] for p in group_positions])
                            extent_x = positions_array[:, 0].max() - positions_array[:, 0].min()
                            extent_y = positions_array[:, 1].max() - positions_array[:, 1].min()
                            
                            # Limit extent to reasonable values to prevent huge ridges
                            extent_x = min(extent_x, 50.0)  # Max 50m extent
                            extent_y = min(extent_y, 50.0)
                            
                            ridge_width = max(extent_x + 10.0, 20.0)  # Add margin, min 20m
                            ridge_length = max(extent_y + 10.0, 30.0)  # Min 30m
                            
                            # Cap maximum size
                            MAX_RIDGE_SIZE = 100.0  # meters
                            ridge_width = min(ridge_width, MAX_RIDGE_SIZE)
                            ridge_length = min(ridge_length, MAX_RIDGE_SIZE)
                            
                            # Check if we already have a ridge zone nearby
                            new_ridge_needed = True
                            for existing_ridge in ridge_zones:
                                if (existing_ridge.position - ridge_center).length < 30:
                                    new_ridge_needed = False
                                    break
                            
                            if new_ridge_needed:
                                new_ridge = create_ridge_zone(
                                    space, ridge_center, ridge_direction, 
                                    width=max(ridge_width, 20.0), 
                                    length=max(ridge_length, 30.0),
                                    creation_time=t,
                                    contributing_floes=compressed_group
                                )
                                ridge_zones.append(new_ridge)
                                
                                # Track ridge creation
                                ridge_history.append({
                                    'time': t,
                                    'x': ridge_center.x,
                                    'y': ridge_center.y,
                                    'direction': ridge_direction,
                                    'num_floes': len(compressed_group)
                                })
                    
                    # LEGACY: Fallback to old method if no compression detected but enough individual candidates
                    if len(compressed_groups) == 0 and len(ridging_candidates) >= RIDGE_MIN_FLOES:
                        # Check if we already have a ridge zone nearby
                        new_ridge_needed = True
                        ridge_center = ship_body.position + Vec2d(
                            np.cos(state.psi) * 25,  # 25m ahead of ship
                            np.sin(state.psi) * 25
                        )
                        
                        for existing_ridge in ridge_zones:
                            if (existing_ridge.position - ridge_center).length < 30:
                                new_ridge_needed = False
                                break
                        
                        if new_ridge_needed:
                            new_ridge = create_ridge_zone(
                                space, ridge_center, state.psi, width=20.0, length=30.0,
                                creation_time=t
                            )
                            ridge_zones.append(new_ridge)
                            
                            # Track ridge creation
                            ridge_history.append({
                                'time': t,
                                'x': ridge_center.x,
                                'y': ridge_center.y,
                                'direction': state.psi,
                                'num_floes': len(ridging_candidates)
                            })
                    
                    # Update ridge geometry dynamically (less frequently than thickness)
                    # NOTE: Dynamic shape updates are disabled by default to prevent physics instability
                    # Only logical position/size are updated, not Pymunk shapes
                    ridge_geometry_update_counter += 1
                    if ridge_geometry_update_counter >= 50:  # Update geometry every 50 iterations (~1 second at 0.02s dt)
                        ridge_geometry_update_counter = 0
                        
                        # Update geometry for each ridge (logical only, no Pymunk shape updates)
                        for ridge_zone in ridge_zones:
                            ridge_zone.update_geometry_from_floes(
                                space, polygons, 
                                min_update_interval=1.0, 
                                current_time=t,
                                enable_dynamic_shape=False  # Disabled for stability
                            )
                        
                        # Check for ridge merging (logical only)
                        merged_indices = set()
                        for i, ridge1 in enumerate(ridge_zones):
                            if i in merged_indices:
                                continue
                            for j, ridge2 in enumerate(ridge_zones[i+1:], start=i+1):
                                if j in merged_indices:
                                    continue
                                if ridge1.can_merge_with(ridge2, merge_distance=40.0):
                                    # Merge ridge2 into ridge1 (logical only)
                                    ridge1.merge_with(ridge2, space, enable_dynamic_shape=False)
                                    merged_indices.add(j)
                        
                        # Remove merged ridges
                        ridge_zones[:] = [r for i, r in enumerate(ridge_zones) if i not in merged_indices]
                    
                    # Update ridge activity and remove decayed ridges
                    ship_position_vec = Vec2d(state.x, state.y)
                    ridges_to_remove = []
                    for i, ridge_zone in enumerate(ridge_zones):
                        # Update activity status
                        ridge_zone.update_activity(ship_position_vec, t, polygons)
                        
                        # Check if ridge should be removed
                        if ridge_zone.should_remove(t):
                            ridges_to_remove.append(i)
                    
                    # Remove decayed ridges (in reverse order to maintain indices)
                    for i in reversed(ridges_to_remove):
                        ridge_zone = ridge_zones[i]
                        # Remove Pymunk shape
                        if ridge_zone.shape is not None:
                            space.remove(ridge_zone.shape)
                        if ridge_zone.body is not None:
                            space.remove(ridge_zone.body)
                        # Remove from list
                        ridge_zones.pop(i)
                    
                    # Update ridge thickness based on nearby floes with compression forces
                    thicknesses = []
                    for ridge_zone in ridge_zones:
                        accumulated_mass = 0.0
                        nearby_floes = []
                        
                        # Find nearby floes and accumulate mass
                        for poly in polygons:
                            distance = (poly.body.position - ridge_zone.position).length
                            if distance < ridge_zone.width:
                                nearby_floes.append(poly)
                                # Enhanced accumulation: rate depends on compression and ice concentration
                                # Higher compression = faster accumulation
                                compression_factor = 1.0 + min(ridge_zone.average_compression_force / 5000.0, 1.0)
                                accumulation_rate = RIDGE_ACCUMULATION_RATE * compression_factor
                                accumulated_mass += poly.mass * accumulation_rate * dt
                        
                        # Compute compression force from contributing floes or nearby floes
                        compression_force = 0.0
                        if len(ridge_zone.contributing_floes) >= 2:
                            # Use contributing floes if available
                            compression_force = compute_compression_force(ridge_zone.contributing_floes)
                        elif len(nearby_floes) >= 2:
                            # Use nearby floes as fallback
                            compression_force = compute_compression_force(nearby_floes)
                        
                        # Update thickness with compression force
                        ridge_zone.update_thickness(accumulated_mass, current_time=t, compression_force=compression_force)
                        thicknesses.append(ridge_zone.thickness)
                    
                    # Track thickness updates
                    if len(ridge_zones) > 0:
                        ridge_thickness_history.append({
                            'time': t,
                            'thickness': thicknesses.copy(),
                            'num_ridges': len(ridge_zones)
                        })
                except Exception as e:
                    # Don't let ridging errors stop the simulation
                    if debug:
                        print(f"Ridging error (non-fatal): {e}")
                        import traceback
                        traceback.print_exc()
                    pass
            
            # clear collision metrics
            system_ke_loss.clear(), delta_ke_ice.clear(), total_impulse.clear(), collided_ob_idx.clear(), contact_pts.clear()

            # move simulation forward
            for _ in range(steps):
                space.step(dt / steps)

            iteration += 1  # increment simulation step

    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, exiting...')

    finally:
        print('\nDone simulation... Processing simulation data...')

        conn_recv.close()
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put(None)
        queue.close()

        if plot is not None:
            plot.close()

        # get logged simulation data
        sim_data = sim_dynamics.get_state_history()

        # make a plot showing the final state of the simulation
        plot = Plot(
            obstacles=get_global_obs_coords(poly_vertices, batched_data[:, :2], batched_data[:, 2]),
            path=path.T, goal=goal[1], map_figsize=None, remove_sim_ticks=False, show=False,
            ship_pos=sim_data[['x', 'y']].to_numpy().T, map_shape=cfg.map_shape, legend=False
        )
        plot.add_ship_patch(plot.sim_ax, cfg.ship.vertices, *state.eta)
        plot.save(save_fig_dir, TRIAL_SIM_PLOT)

        # plots
        collided_ob_idx = sim_data['collided_ob_idx']
        if not np.sum(collided_ob_idx):
            print('No collisions occurred in simulation, skipping collision stats...')
        else:
            floe_mass_hist_plot(
                sim_data, floe_masses,
                save_fig=os.path.join(save_fig_dir, FLOE_MASS_HIST_PLOT) if save_fig_dir else None)
            ke_impulse_vs_time_plot(
                sim_data,
                save_fig=os.path.join(save_fig_dir, KE_IMPULSE_VS_TIME_PLOT) if save_fig_dir else None)
            impact_locs_impulse_plot(
                sim_data, cfg.ship.vertices,
                save_fig=os.path.join(save_fig_dir, IMPACT_LOCS_IMPULSE_PLOT) if save_fig_dir else None)

        state_vs_time_plot(
            sim_data,
            save_fig=os.path.join(save_fig_dir, STATE_VS_TIME_PLOT) if save_fig_dir else None
        )
        control_vs_time_plot(
            sim_data,
            dim_U=len(sim_dynamics.state.u_control),
            control_labels=sim_dynamics.vessel_model.controls,
            save_fig=os.path.join(save_fig_dir, CONTROL_VS_TIME_PLOT) if save_fig_dir else None
        )
        
        # Fracturing and ridging plots
        if FRACTURE_ENABLED:
            if save_fig_dir:
                print(f'Generating fracturing plot... (found {len(fracture_history)} fracture events)')
            fracturing_stats_plot(
                fracture_history,
                save_fig=os.path.join(save_fig_dir, FRACTURING_PLOT) if save_fig_dir else None
            )
        
        if RIDGING_ENABLED:
            if save_fig_dir:
                print(f'Generating ridging plot... (found {len(ridge_history)} ridges, {len(ridge_resistance_history)} resistance samples)')
            ridging_stats_plot(
                ridge_history,
                ridge_thickness_history,
                ridge_resistance_history,
                save_fig=os.path.join(save_fig_dir, RIDGING_PLOT) if save_fig_dir else None
            )

        if plt.get_backend() != 'agg':
            plt.show()
        plt.close('all')

        print('Clean exiting planner process...')
        planner.terminate()
        if planner.is_alive():
            planner.join(timeout=2)

        print('Done')
