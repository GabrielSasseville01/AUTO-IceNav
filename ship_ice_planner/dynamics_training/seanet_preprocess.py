import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import xarray as xr
import pickle
import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. Load monthly ECCO velocity fields (surface layer k=0)
# -------------------------------------------------------------------------

data_dir = Path(__file__).resolve().parent / "data" / "ECCO4"
data_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# WebDAV credentials & configuration
# -------------------------------------------------------------------------
# Set these manually or export the matching environment variables before running.
# WEBDAV_USERNAME = os.environ.get("ECCO_WEBDAV_USERNAME", "")
# WEBDAV_PASSWORD = os.environ.get("ECCO_WEBDAV_PASSWORD", "")
WEBDAV_USERNAME = ""
WEBDAV_PASSWORD = ""

download_files = False

WEBDAV_BASE_URL = os.environ.get(
    "ECCO_WEBDAV_BASE_URL",
    "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly",
)
DEFAULT_YEARS = tuple(range(1992, 2017))
MAX_CONCURRENT_DOWNLOADS = 12


def download_monthly_velocity(
    component: str,
    *,
    years: Iterable[int] = DEFAULT_YEARS,
    months: Iterable[int] = range(1, 13),
    overwrite: bool = False,
) -> None:
    """Download ECCO monthly velocity fields via WebDAV for the given component."""
    if not WEBDAV_USERNAME or not WEBDAV_PASSWORD:
        raise RuntimeError(
            "WebDAV credentials are not set. Update WEBDAV_USERNAME/WEBDAV_PASSWORD."
        )

    component = component.upper()
    base_url = f"{WEBDAV_BASE_URL.rstrip('/')}/{component}"

    download_jobs = []
    for year in years:
        for month in months:
            filename = f"{component}_{year}_{month:02d}.nc"
            url = f"{base_url}/{year}/{filename}"
            local_path = data_dir / filename
            if local_path.exists() and not overwrite:
                continue
            download_jobs.append((url, local_path))

    if not download_jobs:
        print(f"All requested {component} files already exist; skipping download.")
        return

    def _download_job(job):
        url, local_path = job
        print(f"Downloading {url} -> {local_path}")
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")

        with requests.Session() as thread_session:
            thread_session.auth = (WEBDAV_USERNAME, WEBDAV_PASSWORD)
            with thread_session.get(url, stream=True, timeout=60) as response:
                if response.status_code == 404:
                    print(f"Remote file missing: {url}")
                    return
                response.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        tmp_path.replace(local_path)

    max_workers = min(MAX_CONCURRENT_DOWNLOADS, len(download_jobs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_job, job): job for job in download_jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - best effort logging
                url, local_path = job
                print(f"Failed download {url} -> {local_path}: {exc}")
                raise


if download_files and WEBDAV_USERNAME and WEBDAV_PASSWORD:
    download_monthly_velocity("EVEL")
    download_monthly_velocity("NVEL")
else:
    print(
        "WebDAV credentials not set; skipping remote download. "
        "Set WEBDAV_USERNAME/WEBDAV_PASSWORD to enable."
    )

def _list_local(pattern):
    """Return a sorted list of local NetCDF files inside data/ECCO4."""
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files found matching '{pattern}' in {data_dir}")
    return [str(p) for p in matches]

u_files = _list_local("EVEL_*.nc")
v_files = _list_local("NVEL_*.nc")

ds_u = xr.open_mfdataset(u_files, combine="by_coords")
ds_v = xr.open_mfdataset(v_files, combine="by_coords")

# Extract variables (names differ slightly; autodetect)
u_name = [k for k in ds_u.data_vars][0]  # e.g., 'east_vel'
v_name = [k for k in ds_v.data_vars][0]  # e.g., 'north_vel'

U = ds_u[u_name].isel(k=0)  # surface layer
V = ds_v[v_name].isel(k=0)

print("Loaded U and V with shapes:", U.shape, V.shape)
print("coords:", list(U.coords))



# -------------------------------------------------------------------------
# 2. Select a region (example: North Atlantic)
#    Modify as needed:
#    lon_range = (-70, 0)
#    lat_range = (20, 60)
# -------------------------------------------------------------------------

OFFSET = 100
START_LAT = 85.0
START_LON = 0.05
lat_range = (START_LAT, START_LAT + OFFSET)
lon_range = (START_LON, START_LON + OFFSET)

def _sel_region(field, lon_range, lat_range):
    lon_name = "lon" if "lon" in field.coords else "longitude"
    lat_name = "lat" if "lat" in field.coords else "latitude"
    if lon_name not in field.dims:
        lon_dim = field.coords[lon_name].dims[0]
        field = field.swap_dims({lon_dim: lon_name})
    if lat_name not in field.dims:
        lat_dim = field.coords[lat_name].dims[0]
        field = field.swap_dims({lat_dim: lat_name})
    return field.sel({lon_name: slice(*lon_range), lat_name: slice(*lat_range)})

U_reg = _sel_region(U, lon_range, lat_range)
V_reg = _sel_region(V, lon_range, lat_range)

print("Regional domain:", U_reg.shape)

# -------------------------------------------------------------------------
# Apply spatial upsampling using trilinear interpolation
# -------------------------------------------------------------------------
TARGET_HEIGHT = 600
TARGET_WIDTH = 600

def upsample_spatial(field, target_h, target_w):
    """Upsample spatial dimensions using trilinear interpolation.
    
    Args:
        field: xarray DataArray with dimensions (time, lat, lon)
        target_h: target height (lat dimension)
        target_w: target width (lon dimension)
    
    Returns:
        Upsampled numpy array with shape (time, target_h, target_w)
    """
    # Convert to numpy and then to torch tensor
    data = np.asarray(field.values)  # (time, lat, lon)
    data_tensor = torch.from_numpy(data).float()
    
    # Add batch and channel dimensions for interpolate: (1, 1, T, H, W)
    data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
    
    # Get current dimensions
    current_t = data_tensor.shape[2]
    current_h = data_tensor.shape[3]
    current_w = data_tensor.shape[4]
    
    # Apply trilinear interpolation
    upsampled = F.interpolate(
        data_tensor,
        size=(current_t, target_h, target_w),
        mode='trilinear',
        align_corners=False
    )
    
    # Remove batch and channel dimensions: (T, H, W)
    upsampled = upsampled.squeeze(0).squeeze(0)
    
    return upsampled.numpy()

print(f"Upsampling U_reg to ({TARGET_HEIGHT}, {TARGET_WIDTH})...")
U_reg_upsampled = upsample_spatial(U_reg, TARGET_HEIGHT, TARGET_WIDTH)
print(f"Upsampling V_reg to ({TARGET_HEIGHT}, {TARGET_WIDTH})...")
V_reg_upsampled = upsample_spatial(V_reg, TARGET_HEIGHT, TARGET_WIDTH)

print(f"Upsampled U shape: {U_reg_upsampled.shape}")
print(f"Upsampled V shape: {V_reg_upsampled.shape}")

# -------------------------------------------------------------------------
# 3. Convert monthly fields into a simple dynamical model
#    Here: x_t = [u, v]
#    Use persistence + tendency estimated from finite differences.
# -------------------------------------------------------------------------

# Compute approximate monthly acceleration using upsampled data
print("Computing differences...")
# Convert to torch for diff operation
U_tensor = torch.from_numpy(U_reg_upsampled)
V_tensor = torch.from_numpy(V_reg_upsampled)

# Compute differences along time dimension
du_dt = U_tensor[1:] - U_tensor[:-1]  # (T-1, H, W)
dv_dt = V_tensor[1:] - V_tensor[:-1]  # (T-1, H, W)

# Align shapes
U_mid = U_tensor[1:]  # (T-1, H, W)
V_mid = V_tensor[1:]  # (T-1, H, W)

# Stack observations into a single tensor: (time, channels, lat, lon)
print("Formatting tensor...")
obs_tensor = np.stack(
    [
        U_mid.numpy(),
        V_mid.numpy(),
        du_dt.numpy(),
        dv_dt.numpy(),
    ],
    axis=1,
)

print("Observation tensor shape (time, channels, lat, lon):", obs_tensor.shape)

with open("ECCO4.pkl", "wb") as f:
    pickle.dump(obs_tensor, f)
    f.close()