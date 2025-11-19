#!/usr/bin/env python
"""
Wrapper script to run the demo without GUI issues.
Sets matplotlib to use non-GUI backend before importing anything.
"""
import os
# Set matplotlib backend BEFORE importing matplotlib or any plotting code
os.environ['MPLBACKEND'] = 'Agg'

# Now import and run the demo
import sys
from demo_sim2d_ship_ice_navigation import demo

if __name__ == '__main__':
    # Parse arguments (simplified version)
    import argparse
    
    parser = argparse.ArgumentParser(description='Ship ice navigation demo (no GUI)')
    parser.add_argument('exp_config_file', nargs='?', type=str, 
                        default='data/experiment_configs.pkl')
    parser.add_argument('planner_config_file', nargs='?', type=str, 
                        default='configs/sim2d_config.yaml')
    parser.add_argument('-c', dest='ice_concentration', type=float, default=0.5)
    parser.add_argument('-i', dest='ice_field_idx', type=int, default=1)
    parser.add_argument('-s', '--start', nargs=3, metavar=('x', 'y', 'psi'), type=float, default=None)
    parser.add_argument('-g', '--goal', nargs=2, metavar=('x', 'y'), type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-l', '--logging', action='store_true')
    parser.add_argument('-ll', '--log_level', type=int, default=10)
    
    args = parser.parse_args()
    
    print('Running with non-GUI backend (plots will be saved, not displayed)')
    
    demo(cfg_file=args.planner_config_file,
         exp_config_file=args.exp_config_file,
         ice_concentration=args.ice_concentration,
         ice_field_idx=args.ice_field_idx,
         start=args.start,
         goal=args.goal,
         show_anim=False,  # Always disable animation
         output_dir=args.output_dir,
         debug=args.debug,
         logging=args.logging,
         log_level=args.log_level)

