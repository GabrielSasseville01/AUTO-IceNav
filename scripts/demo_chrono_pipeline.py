#!/usr/bin/env python
"""
Demo: Full Chrono/Bonded-DEM Pipeline

This script demonstrates the complete workflow:
1. Load ice field from satellite-derived data (same as sim2d.py uses)
2. Convert polygons to Chrono bonded DEM representation
3. Run ship navigation with physics simulation
4. Visualize fracturing and ice field dynamics

Usage:
    python scripts/demo_chrono_pipeline.py --config configs/sim2d_config.yaml
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import time
import os

# Import the new Chrono backend
from ship_ice_planner.physics.chrono_backend import ChronoIceSimulator
from ship_ice_planner.physics.sim_adapter import ChronoSimAdapter
from ship_ice_planner.physics.ice_material import IceMaterialParams
from ship_ice_planner.physics.visualization.particle_renderer import ParticleRenderer
from ship_ice_planner.physics.visualization.fracture_visualizer import FractureVisualizer
from ship_ice_planner.utils.utils import DotDict
from ship_ice_planner.geometry.polygon import poly_area


def load_ice_field_data(config):
    """Load ice field polygons from pickle file (same source as sim2d.py)."""
    map_file = config.get('map_file', 'data/ice_floes_MEDEA_fram_20100629.pkl')
    
    print(f"Loading ice field from: {map_file}")
    with open(map_file, 'rb') as f:
        obs_dicts = pickle.load(f)
    
    print(f"Loaded {len(obs_dicts)} ice floe polygons")
    
    # Extract vertices and compute properties
    polygons = []
    thicknesses = []
    areas = []
    
    for obs in obs_dicts:
        verts = np.array(obs['vertices'])
        polygons.append(verts)
        thicknesses.append(obs.get('thickness', 1.0))
        areas.append(poly_area(verts))
    
    return polygons, thicknesses, areas, obs_dicts


def visualize_ice_field_comparison(obs_dicts, chrono_adapter, config, save_path=None):
    """
    Side-by-side visualization: Original polygons vs Chrono bonded DEM representation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Get map bounds
    map_shape = config.get('map_shape', [600, 600])
    
    # === Left: Original polygon representation ===
    ax_orig = axes[0]
    ax_orig.set_title('Original Ice Field (Polygon Representation)', fontsize=14)
    
    patches = []
    colors = []
    for obs in obs_dicts:
        verts = np.array(obs['vertices'])
        poly = MplPolygon(verts, closed=True)
        patches.append(poly)
        # Color by area
        area = poly_area(verts)
        colors.append(np.log10(area + 1))
    
    collection = PatchCollection(patches, cmap='Blues', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
    collection.set_array(np.array(colors))
    ax_orig.add_collection(collection)
    
    ax_orig.set_xlim(0, map_shape[0])
    ax_orig.set_ylim(0, map_shape[1])
    ax_orig.set_aspect('equal')
    ax_orig.set_xlabel('X (m)')
    ax_orig.set_ylabel('Y (m)')
    
    # === Right: Chrono bonded DEM representation ===
    ax_chrono = axes[1]
    ax_chrono.set_title('Chrono Backend (Bonded DEM for large floes)', fontsize=14)
    
    # Get floe states from Chrono
    rigid_patches = []
    bonded_count = 0
    
    for floe_id, floe in chrono_adapter.simulator.ice_floes.items():
        if floe.is_bonded and floe.assembly is not None:
            # Draw bonded floe particles
            bonded_count += 1
            for particle in floe.assembly.particles:
                circle = plt.Circle(particle.pos[:2], particle.radius, 
                                   alpha=0.6, color='coral', edgecolor='darkred', linewidth=0.3)
                ax_chrono.add_patch(circle)
            
            # Draw original boundary
            boundary = MplPolygon(floe.vertices[:, :2], fill=False, 
                                 edgecolor='red', linewidth=2, linestyle='--')
            ax_chrono.add_patch(boundary)
        else:
            # Draw rigid floe as polygon
            poly = MplPolygon(floe.vertices[:, :2], closed=True, 
                            alpha=0.7, facecolor='lightblue', edgecolor='blue', linewidth=0.5)
            rigid_patches.append(poly)
            ax_chrono.add_patch(poly)
    
    ax_chrono.set_xlim(0, map_shape[0])
    ax_chrono.set_ylim(0, map_shape[1])
    ax_chrono.set_aspect('equal')
    ax_chrono.set_xlabel('X (m)')
    ax_chrono.set_ylabel('Y (m)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', 
               markersize=10, label=f'Bonded DEM particles ({bonded_count} floes)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue',
               markersize=10, label=f'Rigid floes ({len(chrono_adapter.simulator.ice_floes) - bonded_count})'),
    ]
    ax_chrono.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return fig


def run_ship_collision_demo(chrono_adapter, config, duration=30.0, save_path=None):
    """
    Run a ship collision simulation and visualize ice fracturing.
    """
    print("\n" + "="*60)
    print("Running Ship-Ice Collision Demo with Chrono Backend")
    print("="*60)
    
    # Setup ship
    ship_cfg = config.get('ship', {})
    start_pos = ship_cfg.get('start_pos', [50, 300, 0])  # x, y, heading
    
    chrono_adapter.set_ship_pose(start_pos[0], start_pos[1], start_pos[2])
    print(f"Ship starting at: ({start_pos[0]:.1f}, {start_pos[1]:.1f}), heading: {start_pos[2]:.2f} rad")
    
    # Simulation parameters
    dt = config.get('sim_dynamics', {}).get('dt', 0.02)
    steps = int(duration / dt)
    
    # Recording for visualization
    ship_trajectory = []
    fracture_events = []
    time_history = []
    ke_history = []
    
    # Thrust profile: accelerate towards ice field
    thrust = 1e6  # N
    rudder = 0.0  # straight ahead
    
    print(f"\nSimulating {duration}s ({steps} steps) with dt={dt}s...")
    print(f"Applying thrust: {thrust/1e6:.1f} MN")
    
    start_time = time.time()
    
    for step in range(steps):
        # Apply control
        chrono_adapter.apply_control(thrust, rudder)
        
        # Step simulation
        chrono_adapter.step()
        
        # Record state
        state = chrono_adapter.get_ship_state()
        ship_trajectory.append(state[:2].copy())
        time_history.append(step * dt)
        ke_history.append(chrono_adapter.get_energy_state()['kinetic_energy'])
        
        # Check for fracture events
        events = chrono_adapter.get_fracture_events()
        if events:
            for e in events:
                fracture_events.append(e)
                print(f"  [t={e.get('time', step*dt):.2f}s] FRACTURE at ({e.get('x', 0):.1f}, {e.get('y', 0):.1f})")
        
        # Progress update
        if step % 100 == 0:
            print(f"  Step {step}/{steps} | Ship pos: ({state[0]:.1f}, {state[1]:.1f}) | KE: {ke_history[-1]:.2e} J")
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f}s (realtime factor: {duration/elapsed:.1f}x)")
    print(f"Total fracture events: {len(fracture_events)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # === Top-left: Ship trajectory over ice field ===
    ax = axes[0, 0]
    ax.set_title('Ship Trajectory Through Ice Field', fontsize=12)
    
    map_shape = config.get('map_shape', [600, 600])
    
    # Draw ice floes
    for floe_id, floe in chrono_adapter.simulator.ice_floes.items():
        color = 'coral' if floe.is_bonded else 'lightblue'
        poly = MplPolygon(floe.vertices[:, :2], closed=True, 
                        alpha=0.5, facecolor=color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(poly)
    
    # Draw ship trajectory
    traj = np.array(ship_trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Ship path')
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'r^', markersize=12, label='End')
    
    # Mark fracture locations
    if fracture_events:
        fx = [e.get('x', 0) for e in fracture_events]
        fy = [e.get('y', 0) for e in fracture_events]
        ax.scatter(fx, fy, c='yellow', s=100, marker='*', edgecolors='red', 
                  linewidths=2, label='Fractures', zorder=5)
    
    ax.set_xlim(0, map_shape[0])
    ax.set_ylim(0, map_shape[1])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # === Top-right: Kinetic energy over time ===
    ax = axes[0, 1]
    ax.set_title('System Kinetic Energy', fontsize=12)
    ax.plot(time_history, ke_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kinetic Energy (J)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Mark fracture events
    for event in fracture_events:
        ax.axvline(event.get('time', 0), color='red', alpha=0.3, linestyle='--')
    
    # === Bottom-left: Ship position over time ===
    ax = axes[1, 0]
    ax.set_title('Ship Position vs Time', fontsize=12)
    ax.plot(time_history, traj[:, 0], 'b-', label='X position', linewidth=1.5)
    ax.plot(time_history, traj[:, 1], 'r-', label='Y position', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Bottom-right: Detail view of fracture area ===
    ax = axes[1, 1]
    ax.set_title('Detail: Bonded DEM Fracture Patterns', fontsize=12)
    
    # Find a bonded floe to show in detail
    detail_shown = False
    for floe_id, floe in chrono_adapter.simulator.ice_floes.items():
        if floe.is_bonded and floe.assembly is not None:
            # Draw particles
            for particle in floe.assembly.particles:
                circle = plt.Circle(particle.pos[:2], particle.radius, 
                                   alpha=0.7, color='lightblue', edgecolor='blue', linewidth=0.3)
                ax.add_patch(circle)
            
            # Draw bonds (intact = blue, broken = red)
            for bond in floe.assembly.bonds:
                p1 = floe.assembly.particles[bond.i]
                p2 = floe.assembly.particles[bond.j]
                color = 'red' if bond.broken else 'steelblue'
                linestyle = '--' if bond.broken else '-'
                alpha = 0.8 if bond.broken else 0.4
                ax.plot([p1.pos[0], p2.pos[0]], [p1.pos[1], p2.pos[1]], 
                       color=color, linestyle=linestyle, alpha=alpha, linewidth=1)
            
            # Show boundary
            boundary = MplPolygon(floe.vertices[:, :2], fill=False, 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(boundary)
            
            ax.set_title(f'Bonded Floe Detail: {floe.assembly.n_particles} particles, '
                        f'{floe.assembly.n_broken_bonds}/{floe.assembly.n_bonds} bonds broken', fontsize=11)
            ax.set_aspect('equal')
            ax.autoscale_view()
            detail_shown = True
            break
    
    if not detail_shown:
        ax.text(0.5, 0.5, 'No bonded floes in view', transform=ax.transAxes, 
               ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved simulation results to {save_path}")
    
    return {
        'trajectory': np.array(ship_trajectory),
        'fracture_events': fracture_events,
        'time': np.array(time_history),
        'kinetic_energy': np.array(ke_history),
    }


def main():
    parser = argparse.ArgumentParser(description='Demo Chrono/Bonded-DEM Pipeline')
    parser.add_argument('--config', type=str, default='configs/sim2d_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--duration', type=float, default=20.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (just save)')
    args = parser.parse_args()
    
    # Load config
    print("="*60)
    print("Chrono/Bonded-DEM Pipeline Demo")
    print("="*60)
    print(f"\nLoading config from: {args.config}")
    
    cfg = DotDict.load_from_file(args.config)
    
    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load ice field data
    print("\n[Step 1] Loading ice field data...")
    polygons, thicknesses, areas, obs_dicts = load_ice_field_data(cfg)
    
    # Print statistics
    total_area = sum(areas)
    large_floes = sum(1 for a in areas if a > 500)
    print(f"  Total floes: {len(polygons)}")
    print(f"  Total ice area: {total_area:.0f} m²")
    print(f"  Large floes (>500m², will use bonded DEM): {large_floes}")
    print(f"  Small floes (rigid bodies): {len(polygons) - large_floes}")
    
    # Step 2: Create Chrono simulation
    print("\n[Step 2] Initializing Chrono backend...")
    
    # Add physics config if not present
    if 'physics' not in cfg:
        cfg['physics'] = {'backend': 'chrono', 'use_gpu': False}
    if 'ice_material' not in cfg:
        cfg['ice_material'] = {
            'density': 920,
            'youngs_modulus': 5e9,
            'tensile_strength': 0.5e6,
            'shear_strength': 0.4e6
        }
    if 'bonded_dem' not in cfg:
        cfg['bonded_dem'] = {
            'particle_size_ratio': 0.08,
            'min_floe_area_for_bonding': 500
        }
    
    chrono_adapter = ChronoSimAdapter(cfg)
    
    # Step 3: Load ice field into Chrono
    print("\n[Step 3] Converting ice field to Chrono representation...")
    chrono_adapter.load_ice_field(polygons, thicknesses)
    
    n_bonded = sum(1 for f in chrono_adapter.simulator.ice_floes.values() if f.is_bonded)
    n_rigid = len(chrono_adapter.simulator.ice_floes) - n_bonded
    print(f"  Created {n_bonded} bonded DEM floes")
    print(f"  Created {n_rigid} rigid floes")
    
    # Step 4: Visualize comparison
    print("\n[Step 4] Generating ice field comparison visualization...")
    comparison_path = os.path.join(args.output_dir, 'chrono_pipeline_comparison.png')
    visualize_ice_field_comparison(obs_dicts, chrono_adapter, cfg, save_path=comparison_path)
    
    # Step 5: Run collision simulation
    print("\n[Step 5] Running ship collision simulation...")
    results_path = os.path.join(args.output_dir, 'chrono_pipeline_simulation.png')
    results = run_ship_collision_demo(chrono_adapter, cfg, 
                                      duration=args.duration, 
                                      save_path=results_path)
    
    # Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {comparison_path}")
    print(f"  - {results_path}")
    
    if not args.no_show:
        plt.show()
    
    # Cleanup
    chrono_adapter.cleanup()
    
    return results


if __name__ == '__main__':
    main()
