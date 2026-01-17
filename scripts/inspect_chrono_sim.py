#!/usr/bin/env python
"""
Quick visual inspection of Chrono physics implementation.

Usage:
    python scripts/inspect_chrono_sim.py --mode assembly
    python scripts/inspect_chrono_sim.py --mode fracture --save fracture_test.png
    python scripts/inspect_chrono_sim.py --mode collision
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, '.')

from ship_ice_planner.physics import ChronoIceSimulator, IceMaterialParams, BondedAssembly
from ship_ice_planner.physics.visualization import ParticleRenderer, FractureVisualizer


def inspect_assembly(save_path=None):
    """Show bonded particle assembly structure."""
    print("Creating bonded particle assembly...")
    
    # Create a 20x20 meter square floe
    vertices = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
    material = IceMaterialParams()
    
    # Create assembly
    assembly = BondedAssembly(vertices, material=material)
    
    print(f"Assembly created:")
    print(f"  - Particles: {assembly.n_particles}")
    print(f"  - Bonds: {assembly.n_bonds}")
    print(f"  - Coverage ratio: {assembly.coverage_ratio():.2%}")
    print(f"  - Particle radius: {assembly.particle_radius:.2f} m")
    
    # Visualize
    renderer = ParticleRenderer(figsize=(10, 10))
    renderer.draw_assembly(assembly, color_by='none')
    
    if save_path:
        renderer.save(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
    return assembly


def inspect_fracture(save_path=None):
    """Show fracture pattern after simulated impact."""
    print("Creating bonded floe and simulating fracture...")
    
    # Create a larger floe for more particles
    vertices = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])
    material = IceMaterialParams(tensile_strength=0.5e6, shear_strength=0.4e6)
    
    assembly = BondedAssembly(vertices, material=material, particle_radius=1.0)
    
    print(f"Initial assembly: {assembly.n_particles} particles, {assembly.n_bonds} bonds")
    
    # Simulate impact by breaking bonds near the impact point
    # Impact from the right side (x = 30)
    impact_x = 28
    impact_y = 15
    impact_radius = 8
    
    bonds_broken = 0
    for bond in assembly.bonds:
        pi = assembly.particles[bond.i]
        pj = assembly.particles[bond.j]
        
        # Check if bond is near impact point
        midpoint = (pi.pos + pj.pos) / 2
        dist_to_impact = np.sqrt((midpoint[0] - impact_x)**2 + (midpoint[1] - impact_y)**2)
        
        if dist_to_impact < impact_radius:
            # Break bond with probability decreasing with distance
            break_prob = 1.0 - (dist_to_impact / impact_radius)
            if np.random.random() < break_prob:
                bond.broken = True
                bond.break_mode = 'tensile' if np.random.random() < 0.6 else 'shear'
                bonds_broken += 1
                
    assembly.mark_fragments_dirty()
    
    print(f"Simulated impact: {bonds_broken} bonds broken")
    print(f"Fragments created: {len(assembly.get_fragments())}")
    
    # Visualize
    renderer = ParticleRenderer(figsize=(12, 10))
    renderer.draw_assembly(assembly, show_broken_bonds=True, color_by='fragment')
    
    # Mark impact point
    renderer.ax.scatter([impact_x], [impact_y], c='red', s=200, marker='x', 
                        linewidths=3, label='Impact point')
    renderer.ax.legend()
    
    if save_path:
        renderer.save(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
    return assembly


def inspect_simulator(save_path=None):
    """Test the full Chrono simulator."""
    print("Testing ChronoIceSimulator...")
    
    # Create simulator
    material = IceMaterialParams()
    sim = ChronoIceSimulator(material=material, dt=0.02)
    
    # Create ship
    sim.create_ship(position=(50, 50), heading=0)
    print(f"Ship created at (50, 50)")
    
    # Create some floes
    floe1_verts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    floe2_verts = np.array([[100, 100], [130, 100], [130, 130], [100, 130]])
    
    fid1 = sim.create_ice_floe_rigid(floe1_verts)
    fid2 = sim.create_ice_floe_bonded(floe2_verts)
    
    print(f"Rigid floe created: ID={fid1}")
    print(f"Bonded floe created: ID={fid2}, particles={sim.ice_floes[fid2].n_particles}")
    
    # Run some steps
    print("Running simulation...")
    for i in range(100):
        sim.apply_ship_thrust(1e4, 0)
        sim.step()
        
    state = sim.get_ship_state()
    print(f"Ship state after 100 steps:")
    print(f"  Position: ({state['x']:.2f}, {state['y']:.2f})")
    print(f"  Heading: {state['heading']:.2f} rad")
    print(f"  Velocity: ({state['vx']:.2f}, {state['vy']:.2f}) m/s")
    
    # Visualize bonded floe if present
    if fid2 in sim.ice_floes:
        floe = sim.ice_floes[fid2]
        if floe.assembly:
            renderer = ParticleRenderer(figsize=(10, 10))
            renderer.draw_assembly(floe.assembly, color_by='stress')
            
            if save_path:
                renderer.save(save_path)
                print(f"Saved to {save_path}")
            else:
                plt.show()
                
    sim.cleanup()
    print("Simulator cleaned up")


def main():
    parser = argparse.ArgumentParser(description='Inspect Chrono simulation')
    parser.add_argument('--mode', choices=['assembly', 'fracture', 'collision', 'simulator'],
                       default='assembly', help='Visualization mode')
    parser.add_argument('--save', type=str, help='Save figure to path')
    args = parser.parse_args()
    
    print(f"Running in {args.mode} mode...")
    print("=" * 50)
    
    if args.mode == 'assembly':
        inspect_assembly(args.save)
    elif args.mode == 'fracture':
        inspect_fracture(args.save)
    elif args.mode == 'collision':
        # For collision mode, run the simulator test
        inspect_simulator(args.save)
    elif args.mode == 'simulator':
        inspect_simulator(args.save)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
        
    print("=" * 50)
    print("Done!")


if __name__ == '__main__':
    main()
