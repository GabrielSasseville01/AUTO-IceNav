#!/usr/bin/env python3
"""
Demo script for ice fracturing animation.

This script runs a ship simulation with the SubZero-style stress-based
ice fracturing model and saves an animation.
"""
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for saving animation
import matplotlib.pyplot as plt
import os
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ship_ice_planner.sim2d import sim
from ship_ice_planner.utils.utils import DotDict


def main():
    """Run fracturing demo simulation."""
    # Create output directory with timestamp to avoid conflicts
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'fracture_demo_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    cfg_file = 'configs/sim2d_config.yaml'
    cfg = DotDict.load_from_file(cfg_file)
    cfg.cfg_file = cfg_file
    
    # Override config for fracturing demo
    cfg.output_dir = output_dir
    cfg.anim.save = True   # Save animation
    cfg.anim.show = False  # Don't show live (to save properly)
    cfg.anim.fps = 20
    cfg.anim.plot_steps = 5  # Update every 5 steps
    cfg.sim.t_max = 2000   # Shorter for quicker generation
    cfg.sim_dynamics.target_speed = 4.0  # Faster for more impacts
    
    # Load experiment config
    exp_config_file = 'data/experiment_configs.pkl'
    with open(exp_config_file, 'rb') as f:
        exp_config = pickle.load(f)
    
    # Use a moderate concentration for good demo
    ice_concentration = 0.4
    ice_field_idx = 5  # Pick a specific ice field
    
    exp = exp_config['exp'][ice_concentration][ice_field_idx]
    
    print("=" * 60)
    print("Ice Fracturing Demo")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Ice concentration: {ice_concentration}")
    print(f"Ice field index: {ice_field_idx}")
    print("=" * 60)
    print("\nStarting simulation with SubZero-style ice fracturing...")
    print("- Stress tensor computation from contact forces")
    print("- Mohr's yield criterion for fracture decision")
    print("- Voronoi tessellation for floe splitting")
    print("- Corner grinding for sharp angle breaking")
    print("=" * 60)
    
    # Run simulation
    sim(cfg=cfg,
        debug=False,
        logging=False,
        log_level=10,
        init_queue=exp)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print(f"Animation saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
