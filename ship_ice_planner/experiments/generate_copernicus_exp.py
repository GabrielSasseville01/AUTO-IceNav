"""
Generate experiment configurations from Copernicus sea ice data.

This script downloads and processes Copernicus Marine Service sea ice data
to create ice field configurations compatible with the simulation.

Usage:
    python -m ship_ice_planner.experiments.generate_copernicus_exp \
        --region "Weddell Sea" \
        --bbox -60 -75 -50 -70 \
        --dates 2024-01-15 2024-01-16 2024-01-17 \
        --output data/copernicus_experiment_configs.pkl
"""

import argparse
import os
import pickle
from typing import List, Tuple
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from ship_ice_planner.utils.copernicus_ice_loader import (
    load_copernicus_ice_data,
    create_demo_copernicus_data
)
from ship_ice_planner.utils.sim_utils import ICE_DENSITY, ICE_THICKNESS


# Predefined regions (bbox in lon, lat)
REGIONS = {
    "Weddell Sea": (-60.0, -75.0, -50.0, -70.0),
    "Beaufort Sea": (-150.0, 70.0, -130.0, 75.0),
    "Barents Sea": (15.0, 70.0, 35.0, 80.0),
    "Kara Sea": (60.0, 70.0, 80.0, 80.0),
    "Chukchi Sea": (-170.0, 65.0, -160.0, 72.0),
}


def generate_copernicus_ice_fields(
    bbox: Tuple[float, float, float, float],
    dates: List[str],
    output_file: str,
    use_api: bool = True,
    data_dir: str = None,
    username: str = None,
    password: str = None,
    min_ice_concentration: float = 0.15,
    min_floe_area: float = 16.0,
    max_floe_area: float = 1e6,
    resolution: float = 100.0,
    use_demo_fallback: bool = True,
    demo_area_size: float = 10000.0
):
    """
    Generate experiment configurations from Copernicus data.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in degrees
        dates: List of date strings in format 'YYYY-MM-DD'
        output_file: Path to output pickle file
        use_api: If True, use Copernicus API; if False, use local files
        data_dir: Directory containing local NetCDF files (if use_api=False)
        username: Copernicus username
        password: Copernicus password
        min_ice_concentration: Minimum ice concentration threshold
        min_floe_area: Minimum floe area in m²
        max_floe_area: Maximum floe area in m²
        resolution: Target resolution in meters per pixel
        use_demo_fallback: If True, use demo data when API fails
        demo_area_size: Size of demo area in meters (default: 10000 = 10km)
    """
    # Convert bbox to local coordinates for ship/goal positioning
    lat_center = (bbox[1] + bbox[3]) / 2
    lon_to_m = 111320 * np.cos(np.deg2rad(lat_center))
    lat_to_m = 111320
    
    width_m = (bbox[2] - bbox[0]) * lon_to_m
    height_m = (bbox[3] - bbox[1]) * lat_to_m
    
    # Default ship start/goal positions will be set per experiment
    # (may be adjusted for demo data vs real data)
    
    exp_config = {
        'exp': {},
        'meta_data': {
            'source': 'Copernicus Marine Service',
            'bbox': bbox,
            'dates': dates,
            'map_shape': (height_m, width_m),
            'resolution': resolution
        }
    }
    
    print(f"Generating ice fields for {len(dates)} dates...")
    print(f"Region bbox: {bbox}")
    print(f"Map size: {width_m:.1f} m × {height_m:.1f} m")
    
    for date_str in tqdm(dates, desc="Processing dates"):
        try:
            if use_api:
                obs_dicts = load_copernicus_ice_data(
                    bbox=bbox,
                    date=date_str,
                    min_ice_concentration=min_ice_concentration,
                    min_floe_area=min_floe_area,
                    max_floe_area=max_floe_area,
                    resolution=resolution,
                    use_api=True,
                    username=username,
                    password=password
                )
            else:
                # Load from local file
                if data_dir is None:
                    raise ValueError("data_dir required when use_api=False")
                data_file = os.path.join(data_dir, f"ice_data_{date_str}.nc")
                if not os.path.exists(data_file):
                    raise FileNotFoundError(f"Data file not found: {data_file}")
                
                obs_dicts = load_copernicus_ice_data(
                    bbox=bbox,
                    date=date_str,
                    min_ice_concentration=min_ice_concentration,
                    min_floe_area=min_floe_area,
                    max_floe_area=max_floe_area,
                    resolution=resolution,
                    use_api=False,
                    data_file=data_file
                )
            
            # Track if we used demo data (to adjust ship/goal positions)
            used_demo_data = False
            if len(obs_dicts) == 0:
                print(f"Warning: No ice floes found for {date_str}")
                if use_demo_fallback:
                    print(f"Using demo data for {date_str}")
                    obs_dicts = create_demo_copernicus_data(bbox, num_floes=50, max_area_m=demo_area_size)
                    used_demo_data = True
                else:
                    continue
            
            # Calculate actual concentration
            # If using demo data, use the demo area size, otherwise use full map area
            if used_demo_data:
                # Demo data is in a limited rectangular area (1000m × 200m to match typical configs)
                demo_length = min(demo_area_size, 1000.0)
                demo_width = min(demo_area_size, 200.0)
                total_map_area = demo_length * demo_width
            else:
                total_map_area = width_m * height_m
            total_ice_area = sum(obs['area'] for obs in obs_dicts)
            concentration = total_ice_area / total_map_area if total_map_area > 0 else 0.0
            
            # Adjust ship/goal positions for demo data
            if used_demo_data:
                # Demo data obstacles are already at (0, 0) origin
                # Use a reasonable size that matches typical config (1100m length, 200m width)
                # Use rectangular area: 1000m length × 200m width to match typical configs
                demo_length = min(demo_area_size, 1000.0)  # Match typical config length
                demo_width = min(demo_area_size, 200.0)    # Match typical config width
                
                # Find actual bounds of obstacles to set ship/goal appropriately
                if len(obs_dicts) > 0:
                    all_vertices = np.vstack([obs['vertices'] for obs in obs_dicts])
                    max_x = all_vertices[:, 0].max()
                    max_y = all_vertices[:, 1].max()
                    # Use actual bounds or demo area, whichever is smaller
                    actual_width = min(max_x, demo_width)
                    actual_length = min(max_y, demo_length)
                else:
                    actual_width = demo_width
                    actual_length = demo_length
                
                # Ship starts at bottom (y=0), goal at top (y=actual_length)
                # Center horizontally (x=actual_width/2)
                ship_start = (actual_width / 2, 0, np.pi / 2)
                goal = (actual_width / 2, actual_length)
            else:
                # Use full map positions
                ship_start = (width_m / 2, 0, np.pi / 2)
                goal = (width_m / 2, height_m)
            
            # Round concentration to nearest 0.1 for grouping
            concentration_key = round(concentration, 1)
            
            # Initialize concentration group if needed
            if concentration_key not in exp_config['exp']:
                exp_config['exp'][concentration_key] = {}
            
            # Find next available index
            if len(exp_config['exp'][concentration_key]) > 0:
                next_idx = max(exp_config['exp'][concentration_key].keys()) + 1
            else:
                next_idx = 0
            
            # Store experiment
            exp_config['exp'][concentration_key][next_idx] = {
                'obstacles': obs_dicts,
                'ship_state': ship_start,
                'goal': goal,
                'date': date_str,
                'concentration': concentration,
                'num_floes': len(obs_dicts)
            }
            
            print(f"  {date_str}: {len(obs_dicts)} floes, "
                  f"concentration={concentration:.2%}")
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            if use_demo_fallback:
                print(f"Using demo data for {date_str}")
                try:
                    # Use reasonable area size to match typical configs (1000m × 200m) and avoid memory issues
                    demo_length = min(demo_area_size, 1000.0)
                    demo_width = min(demo_area_size, 200.0)
                    # Generate in square area, then we'll adjust positions
                    area_size = max(demo_length, demo_width)
                    obs_dicts = create_demo_copernicus_data(bbox, num_floes=50, max_area_m=area_size)
                    
                    # Obstacles are already at (0, 0) from create_demo_copernicus_data
                    # Find actual bounds for ship/goal positioning
                    if len(obs_dicts) > 0:
                        all_vertices = np.vstack([obs['vertices'] for obs in obs_dicts])
                        max_x = all_vertices[:, 0].max()
                        max_y = all_vertices[:, 1].max()
                        actual_width = min(max_x, demo_width)
                        actual_length = min(max_y, demo_length)
                    else:
                        actual_width = demo_width
                        actual_length = demo_length
                    
                    # Calculate concentration for demo area
                    total_ice_area = sum(obs['area'] for obs in obs_dicts)
                    total_map_area = actual_length * actual_width
                    concentration = total_ice_area / total_map_area if total_map_area > 0 else 0.4
                    concentration_key = round(concentration, 1)
                    
                    # Ship/goal positions relative to obstacles (starting from origin)
                    ship_start = (actual_width / 2, 0, np.pi / 2)
                    goal = (actual_width / 2, actual_length)
                    
                    if concentration_key not in exp_config['exp']:
                        exp_config['exp'][concentration_key] = {}
                    
                    next_idx = max(exp_config['exp'][concentration_key].keys()) + 1 if exp_config['exp'][concentration_key] else 0
                    
                    exp_config['exp'][concentration_key][next_idx] = {
                        'obstacles': obs_dicts,
                        'ship_state': ship_start,
                        'goal': goal,
                        'date': date_str,
                        'concentration': concentration,
                        'num_floes': len(obs_dicts)
                    }
                    print(f"  {date_str} (demo): {len(obs_dicts)} floes, "
                          f"concentration={concentration:.2%}")
                except Exception as e2:
                    print(f"Demo data generation also failed: {e2}")
            continue
    
    # Save experiment config
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(exp_config, f)
    
    print(f"\nSaved experiment config to: {output_file}")
    print(f"Generated {sum(len(v) for v in exp_config['exp'].values())} ice fields")
    print(f"Concentrations: {sorted(exp_config['exp'].keys())}")


if __name__ == '__main__':
    import numpy as np
    
    parser = argparse.ArgumentParser(
        description='Generate experiment configurations from Copernicus sea ice data'
    )
    parser.add_argument(
        '--region', type=str, choices=list(REGIONS.keys()),
        help='Predefined region name'
    )
    parser.add_argument(
        '--bbox', nargs=4, type=float, metavar=('min_lon', 'min_lat', 'max_lon', 'max_lat'),
        help='Bounding box in degrees (longitude, latitude)'
    )
    parser.add_argument(
        '--dates', nargs='+', type=str, required=True,
        help='Date(s) in format YYYY-MM-DD'
    )
    parser.add_argument(
        '--output', type=str, default='data/copernicus_experiment_configs.pkl',
        help='Output pickle file path'
    )
    parser.add_argument(
        '--use-api', action='store_true', default=True,
        help='Use Copernicus API (default: True)'
    )
    parser.add_argument(
        '--no-api', dest='use_api', action='store_false',
        help='Use local NetCDF files instead of API'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory containing local NetCDF files (required if --no-api)'
    )
    parser.add_argument(
        '--username', type=str, default=None,
        help='Copernicus username (or set COPERNICUS_USERNAME env var)'
    )
    parser.add_argument(
        '--password', type=str, default=None,
        help='Copernicus password (or set COPERNICUS_PASSWORD env var)'
    )
    parser.add_argument(
        '--min-concentration', type=float, default=0.15,
        help='Minimum ice concentration threshold (default: 0.15)'
    )
    parser.add_argument(
        '--min-floe-area', type=float, default=16.0,
        help='Minimum floe area in m² (default: 16.0)'
    )
    parser.add_argument(
        '--max-floe-area', type=float, default=1e6,
        help='Maximum floe area in m² (default: 1e6)'
    )
    parser.add_argument(
        '--resolution', type=float, default=100.0,
        help='Target resolution in meters per pixel (default: 100.0)'
    )
    parser.add_argument(
        '--demo-area-size', type=float, default=1000.0,
        help='Size of demo area in meters when using demo fallback (default: 1000.0 = 1km)'
    )
    parser.add_argument(
        '--no-demo-fallback', dest='use_demo_fallback', action='store_false',
        help='Do not use demo data when API fails'
    )
    
    args = parser.parse_args()
    
    # Determine bbox
    if args.region:
        bbox = REGIONS[args.region]
        print(f"Using predefined region: {args.region}")
    elif args.bbox:
        bbox = tuple(args.bbox)
    else:
        parser.error("Must specify either --region or --bbox")
    
    # Get credentials
    username = args.username or os.getenv('COPERNICUS_USERNAME')
    password = args.password or os.getenv('COPERNICUS_PASSWORD')
    
    # Generate ice fields
    generate_copernicus_ice_fields(
        bbox=bbox,
        dates=args.dates,
        output_file=args.output,
        use_api=args.use_api,
        data_dir=args.data_dir,
        username=username,
        password=password,
        min_ice_concentration=args.min_concentration,
        min_floe_area=args.min_floe_area,
        max_floe_area=args.max_floe_area,
        resolution=args.resolution,
        use_demo_fallback=args.use_demo_fallback,
        demo_area_size=args.demo_area_size
    )

