"""
Copernicus Marine Service sea ice data loader for simulation.

This module provides functionality to:
1. Fetch sea ice data from Copernicus Marine Service
2. Process concentration grids into individual ice floes
3. Convert to simulation-compatible format (obs_dicts)

Requirements:
- Register at https://marine.copernicus.eu/ for API access
- Install: pip install copernicusmarine xarray netcdf4 scikit-image
"""

import os
import warnings
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, date

import numpy as np
from scipy.interpolate import splprep, splev
from skimage import measure, morphology

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    warnings.warn("xarray not available. Install with: pip install xarray netcdf4")

try:
    from copernicusmarine import describe, open_dataset
    COPERNICUS_AVAILABLE = True
except ImportError:
    COPERNICUS_AVAILABLE = False
    warnings.warn("copernicusmarine not available. Install with: pip install copernicusmarine")

from ship_ice_planner.geometry.polygon import poly_area
from ship_ice_planner.utils.sim_utils import ICE_DENSITY, ICE_THICKNESS


# Default Copernicus dataset IDs (may need to be updated based on available datasets)
# Common sea ice concentration datasets:
# - Arctic: "cmems_mod_arc_phy_anfc_0.027deg-2D_P1D-m" (may not exist)
# - Global: Try searching for "sea_ice_concentration" or "siconc" products
# Note: Dataset IDs change over time. Check https://data.marine.copernicus.eu/products
COPERNICUS_ICE_CONCENTRATION_DATASET = None  # Will need to be set based on available products
COPERNICUS_ICE_THICKNESS_DATASET = None


def load_copernicus_ice_data(
    bbox: Tuple[float, float, float, float],
    date: str,
    min_ice_concentration: float = 0.15,
    min_floe_area: float = 16.0,
    max_floe_area: float = 1e6,
    resolution: float = 100.0,
    use_api: bool = True,
    data_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load Copernicus sea ice data and convert to obstacle format.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in degrees
        date: Date string in format 'YYYY-MM-DD'
        min_ice_concentration: Minimum ice concentration to consider (0-1)
        min_floe_area: Minimum floe area in m² to include
        max_floe_area: Maximum floe area in m² to include
        resolution: Target resolution in meters per pixel
        use_api: If True, use Copernicus API; if False, use local file
        data_file: Path to local NetCDF file (if use_api=False)
        username: Copernicus username (optional, can use env vars)
        password: Copernicus password (optional, can use env vars)
    
    Returns:
        List of obstacle dictionaries compatible with simulation format
    """
    # Load ice concentration data
    if use_api:
        if not COPERNICUS_AVAILABLE:
            raise ImportError("copernicusmarine package required for API access. "
                            "Install with: pip install copernicusmarine")
        ice_data = _fetch_copernicus_data_api(bbox, date, username, password)
    else:
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for local file loading. "
                            "Install with: pip install xarray netcdf4")
        if data_file is None:
            raise ValueError("data_file required when use_api=False")
        ice_data = _load_copernicus_data_file(data_file, bbox, date)
    
    # Process concentration grid into individual floes
    obs_dicts = _segment_ice_concentration_to_floes(
        ice_data,
        bbox,
        min_ice_concentration=min_ice_concentration,
        min_floe_area=min_floe_area,
        max_floe_area=max_floe_area,
        resolution=resolution
    )
    
    return obs_dicts


def _fetch_copernicus_data_api(
    bbox: Tuple[float, float, float, float],
    date: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> xr.Dataset:
    """
    Fetch sea ice data from Copernicus Marine Service API.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        date: Date string 'YYYY-MM-DD'
        username: Optional username (uses env vars if None)
        password: Optional password (uses env vars if None)
    
    Returns:
        xarray Dataset with ice concentration data
    """
    if not COPERNICUS_AVAILABLE:
        raise ImportError("copernicusmarine package not available")
    
    # Get credentials from environment if not provided
    if username is None:
        username = os.getenv('COPERNICUS_USERNAME')
    if password is None:
        password = os.getenv('COPERNICUS_PASSWORD')
    
    if username is None or password is None:
        warnings.warn("Copernicus credentials not found. "
                    "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables, "
                    "or pass username/password arguments.")
    
    # Parse date
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    
    # Open dataset (this is a simplified example - actual dataset IDs may vary)
    # Note: You may need to adjust dataset ID based on available Copernicus products
    if COPERNICUS_ICE_CONCENTRATION_DATASET is None:
        raise ValueError(
            "Copernicus dataset ID not configured. "
            "Please check available datasets at https://data.marine.copernicus.eu/products "
            "and update COPERNICUS_ICE_CONCENTRATION_DATASET in copernicus_ice_loader.py"
        )
    
    try:
        # Try to open ice concentration dataset
        # The actual dataset structure depends on Copernicus product
        # Common variable names: siconc, ice_conc, concentration, sic
        ds = open_dataset(
            dataset_id=COPERNICUS_ICE_CONCENTRATION_DATASET,
            username=username,
            password=password,
            variables=["siconc"],  # Sea ice concentration variable name (may need adjustment)
            minimum_longitude=bbox[0],
            maximum_longitude=bbox[2],
            minimum_latitude=bbox[1],
            maximum_latitude=bbox[3],
            start_datetime=date_obj,
            end_datetime=date_obj
        )
        return ds
    except Exception as e:
        warnings.warn(f"Failed to fetch from API: {e}. "
                     "You may need to check dataset ID or use local file. "
                     "See https://data.marine.copernicus.eu/products for available datasets.")
        raise


def _load_copernicus_data_file(
    data_file: str,
    bbox: Tuple[float, float, float, float],
    date: str
) -> xr.Dataset:
    """
    Load Copernicus data from local NetCDF file.
    
    Args:
        data_file: Path to NetCDF file
        bbox: (min_lon, min_lat, max_lon, max_lat) for subsetting
        date: Date string 'YYYY-MM-DD'
    
    Returns:
        xarray Dataset
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray required for file loading")
    
    ds = xr.open_dataset(data_file)
    
    # Subset by bbox
    ds = ds.sel(
        longitude=slice(bbox[0], bbox[2]),
        latitude=slice(bbox[1], bbox[3])
    )
    
    # Subset by date if time dimension exists
    if 'time' in ds.dims:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        ds = ds.sel(time=date_obj, method='nearest')
    
    return ds


def _segment_ice_concentration_to_floes(
    ice_data: xr.Dataset,
    bbox: Tuple[float, float, float, float],
    min_ice_concentration: float = 0.15,
    min_floe_area: float = 16.0,
    max_floe_area: float = 1e6,
    resolution: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Segment ice concentration grid into individual ice floes.
    
    This function:
    1. Extracts ice concentration array
    2. Thresholds to get ice regions
    3. Segments into individual floes using connected components
    4. Converts each floe to a polygon
    5. Calculates properties (area, mass, center)
    
    Args:
        ice_data: xarray Dataset with ice concentration
        bbox: (min_lon, min_lat, max_lon, max_lat)
        min_ice_concentration: Minimum concentration threshold
        min_floe_area: Minimum floe area in m²
        max_floe_area: Maximum floe area in m²
        resolution: Meters per pixel for area calculation
    
    Returns:
        List of obstacle dictionaries
    """
    # Extract ice concentration variable
    # Common variable names: siconc, ice_conc, concentration, etc.
    conc_var = None
    for var_name in ['siconc', 'ice_conc', 'concentration', 'sic']:
        if var_name in ice_data.data_vars:
            conc_var = ice_data[var_name]
            break
    
    if conc_var is None:
        # Try first data variable
        if len(ice_data.data_vars) > 0:
            conc_var = list(ice_data.data_vars.values())[0]
        else:
            raise ValueError("No ice concentration variable found in dataset")
    
    # Get concentration array (handle time dimension if present)
    if 'time' in conc_var.dims:
        conc_array = conc_var.isel(time=0).values
    else:
        conc_array = conc_var.values
    
    # Handle NaN and fill values
    conc_array = np.nan_to_num(conc_array, nan=0.0)
    
    # Get coordinate arrays
    if 'longitude' in conc_var.coords:
        lon = conc_var.longitude.values
        lat = conc_var.latitude.values
    elif 'lon' in conc_var.coords:
        lon = conc_var.lon.values
        lat = conc_var.lat.values
    else:
        # Create coordinate arrays from bbox and array shape
        lon = np.linspace(bbox[0], bbox[2], conc_array.shape[1])
        lat = np.linspace(bbox[1], bbox[3], conc_array.shape[0])
    
    # Threshold to get ice regions
    ice_mask = conc_array >= min_ice_concentration
    
    # Clean up mask (remove small holes, smooth edges)
    ice_mask = morphology.binary_closing(ice_mask, morphology.disk(2))
    ice_mask = morphology.binary_opening(ice_mask, morphology.disk(1))
    
    # Label connected components (individual floes)
    labeled_floes = measure.label(ice_mask, connectivity=2)
    num_floes = labeled_floes.max()
    
    if num_floes == 0:
        warnings.warn("No ice floes found in data")
        return []
    
    # Convert to local Cartesian coordinates (UTM approximation)
    # For small regions, we can use simple lat/lon to meters conversion
    # For production, use proper UTM projection (e.g., pyproj)
    lat_center = (bbox[1] + bbox[3]) / 2
    lon_to_m = 111320 * np.cos(np.deg2rad(lat_center))  # meters per degree longitude
    lat_to_m = 111320  # meters per degree latitude
    
    obs_dicts = []
    
    # Process each floe
    for floe_id in range(1, num_floes + 1):
        floe_mask = (labeled_floes == floe_id)
        
        # Get floe properties
        props = measure.regionprops(floe_mask.astype(int))[0]
        
        # Calculate area in meters²
        # Area in pixels * resolution²
        pixel_area = props.area
        area_m2 = pixel_area * resolution * resolution
        
        # Filter by area
        if area_m2 < min_floe_area or area_m2 > max_floe_area:
            continue
        
        # Get contour (boundary) of floe
        contours = measure.find_contours(floe_mask.astype(float), 0.5)
        if len(contours) == 0:
            continue
        
        # Use largest contour (outer boundary)
        contour = max(contours, key=len)
        
        # Simplify contour to reduce vertices
        if len(contour) > 3:
            # Fit spline and resample
            tck, u = splprep([contour[:, 1], contour[:, 0]], s=0, k=min(3, len(contour)-1))
            u_new = np.linspace(0, 1, min(50, len(contour)))  # Limit to 50 vertices
            contour_smooth = splev(u_new, tck)
            contour = np.column_stack([contour_smooth[1], contour_smooth[0]])
        
        # Convert contour indices to world coordinates (meters)
        # Contour is in (row, col) format, need to convert to (lat, lon) then to meters
        vertices_lon = np.interp(contour[:, 1], np.arange(len(lon)), lon)
        vertices_lat = np.interp(contour[:, 0], np.arange(len(lat)), lat)
        
        # Convert to local Cartesian (meters) relative to bbox origin
        origin_lon = bbox[0]
        origin_lat = bbox[1]
        vertices_x = (vertices_lon - origin_lon) * lon_to_m
        vertices_y = (vertices_lat - origin_lat) * lat_to_m
        
        vertices = np.column_stack([vertices_x, vertices_y])
        
        # Calculate center
        center_y_idx, center_x_idx = props.centroid
        center_lon = np.interp(center_x_idx, np.arange(len(lon)), lon)
        center_lat = np.interp(center_y_idx, np.arange(len(lat)), lat)
        center_x = (center_lon - origin_lon) * lon_to_m
        center_y = (center_lat - origin_lat) * lat_to_m
        centre = (center_x, center_y)
        
        # Calculate approximate radius
        radius = np.sqrt(area_m2 / np.pi)
        
        # Calculate mass (area × density × thickness)
        mass = area_m2 * ICE_DENSITY * ICE_THICKNESS
        
        # Create obstacle dictionary
        obs_dict = {
            'vertices': vertices,
            'centre': centre,
            'radius': radius,
            'area': area_m2,
            'mass': mass
        }
        
        obs_dicts.append(obs_dict)
    
    return obs_dicts


def create_demo_copernicus_data(
    bbox: Tuple[float, float, float, float],
    num_floes: int = 50,
    concentration: float = 0.4,
    max_area_m: float = 1000.0
) -> List[Dict[str, Any]]:
    """
    Create demo ice data in Copernicus-like format for testing.
    
    This generates synthetic ice floes when real Copernicus data is unavailable.
    Uses a reasonable area size to avoid memory issues.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        num_floes: Number of floes to generate
        concentration: Target ice concentration
        max_area_m: Maximum area dimension in meters (default: 1km to match typical configs)
    
    Returns:
        List of obstacle dictionaries
    """
    from ship_ice_planner.utils.sim_utils import generate_obstacles
    
    # Convert bbox to meters
    lat_center = (bbox[1] + bbox[3]) / 2
    lon_to_m = 111320 * np.cos(np.deg2rad(lat_center))
    lat_to_m = 111320
    
    width_m = (bbox[2] - bbox[0]) * lon_to_m
    height_m = (bbox[3] - bbox[1]) * lat_to_m
    
    # Limit to reasonable area to avoid memory issues
    # Use a square area centered in the region
    area_size = min(max_area_m, width_m, height_m)
    
    # Calculate number of obstacles for target concentration in this area
    target_area = area_size * area_size * concentration
    avg_floe_area = 500.0  # m²
    estimated_num = int(target_area / avg_floe_area)
    num_floes = min(num_floes, estimated_num) if estimated_num > 0 else num_floes
    
    # Generate obstacles in the limited area
    # Center the area within the bbox
    offset_x = (width_m - area_size) / 2
    offset_y = (height_m - area_size) / 2
    
    obs_dicts, _ = generate_obstacles(
        num_obs=num_floes,
        min_r=2,
        max_r=50,
        min_x=0,
        max_x=area_size,
        min_y=0,
        max_y=area_size,
        allow_overlap=False
    )
    
    # Don't translate obstacles - keep them at origin (0, 0) for costmap compatibility
    # The caller will handle positioning if needed
    # Translation removed: obstacles should start from (0, 0) to match costmap expectations
    
    return obs_dicts


# Example usage and testing
if __name__ == "__main__":
    # Example: Load ice data for a region
    # bbox = (min_lon, min_lat, max_lon, max_lat)
    # Example: Weddell Sea region
    bbox = (-60.0, -75.0, -50.0, -70.0)
    date = "2024-01-15"
    
    try:
        # Try to load from API (requires credentials)
        obs_dicts = load_copernicus_ice_data(
            bbox=bbox,
            date=date,
            use_api=True
        )
        print(f"Loaded {len(obs_dicts)} ice floes from Copernicus API")
    except Exception as e:
        print(f"API load failed: {e}")
        print("Using demo data instead...")
        obs_dicts = create_demo_copernicus_data(bbox, num_floes=50)
        print(f"Generated {len(obs_dicts)} demo ice floes")
    
    print(f"First floe: area={obs_dicts[0]['area']:.1f} m², "
          f"mass={obs_dicts[0]['mass']:.1e} kg")

