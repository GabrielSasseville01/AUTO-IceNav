"""
Helper functions for loading and processing ridge costmap files.
"""
import pickle
import logging
import numpy as np
import os
from typing import Optional, Tuple
from skimage import draw

logger = logging.getLogger(__name__)


def load_ridge_costmap(ridge_costmap_file: str) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load ridge costmap from pickle file.
    
    Args:
        ridge_costmap_file: Path to ridge costmap pickle file
        
    Returns:
        Tuple of (ridge_costmap, scale) or None if loading fails
        - ridge_costmap: numpy array of ridge density values (0-1)
        - scale: scale factor (meters per pixel)
    """
    if not ridge_costmap_file or not os.path.exists(ridge_costmap_file):
        logger.warning(f'Ridge costmap file not found: {ridge_costmap_file}')
        return None
    
    try:
        with open(ridge_costmap_file, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different possible formats
        if isinstance(data, dict):
            ridge_costmap = data.get('ridge_costmap', data.get('costmap', None))
            scale = data.get('scale', data.get('costmap_scale', 1.0))
        elif isinstance(data, np.ndarray):
            ridge_costmap = data
            scale = 1.0  # Default scale if not specified
        else:
            logger.error(f'Unexpected ridge costmap file format: {type(data)}')
            return None
        
        if ridge_costmap is None:
            logger.error('Ridge costmap data not found in pickle file')
            return None
        
        logger.info(f'Loaded ridge costmap: shape={ridge_costmap.shape}, scale={scale}')
        return ridge_costmap, scale
        
    except Exception as e:
        logger.warning(f'Failed to load ridge costmap: {e}')
        return None


def resize_ridge_costmap_to_match(
    ridge_costmap: np.ndarray,
    ridge_scale: float,
    target_shape: Tuple[int, int],
    target_scale: float,
    polygon_file: Optional[str] = None,
    padding: int = 0
) -> np.ndarray:
    """
    Resize ridge costmap to match target costmap dimensions.
    
    If polygon_file is provided, maps ridge density to exact ice floe polygon locations.
    Otherwise, uses simple interpolation.
    
    Args:
        ridge_costmap: Original ridge costmap array
        ridge_scale: Original scale (meters per pixel) of ridge costmap
        target_shape: Target (height, width) for output
        target_scale: Target scale (meters per pixel) for output
        polygon_file: Optional path to polygon file for precise mapping
        padding: Padding offset in meters (for coordinate alignment)
        
    Returns:
        Resized ridge costmap matching target_shape
    """
    target_h, target_w = target_shape
    resized = np.zeros(target_shape, dtype=ridge_costmap.dtype)
    
    # Try polygon-based mapping first if polygon file is provided
    if polygon_file and os.path.exists(polygon_file):
        try:
            with open(polygon_file, 'rb') as f:
                polygons = pickle.load(f)
            
            logger.info(f'Mapping ridge density to {len(polygons)} polygons')
            
            # Map ridge density to each polygon
            for poly_idx, poly in enumerate(polygons):
                vertices = poly['vertices']  # Original world coordinates (0-600m)
                
                # Use vertices as-is (no shift) - convert directly to grid coordinates
                grid_vertices = vertices * target_scale
                
                # Get all grid cells inside this polygon
                rr, cc = draw.polygon(grid_vertices[:, 1], grid_vertices[:, 0], shape=target_shape)
                
                # For each grid cell, lookup ridge density
                for r, c in zip(rr, cc):
                    # Convert grid coordinates back to world coordinates
                    world_x = c / target_scale
                    world_y = r / target_scale
                    
                    # Convert to ridge costmap pixel coordinates
                    ridge_x = int(world_x / ridge_scale)
                    ridge_y = int(world_y / ridge_scale)
                    
                    # Check bounds and assign ridge density
                    if 0 <= ridge_y < ridge_costmap.shape[0] and 0 <= ridge_x < ridge_costmap.shape[1]:
                        if 0 <= r < target_h and 0 <= c < target_w:
                            resized[r, c] = ridge_costmap[ridge_y, ridge_x]
            
            logger.info(f'Mapped ridge density to {np.count_nonzero(resized)} grid cells')
            return resized
            
        except Exception as e:
            logger.warning(f'Error mapping ridge density to polygons: {e}, using simple resize')
    
    # Fallback to simple interpolation
    import cv2
    interpolation = cv2.INTER_AREA if (ridge_costmap.shape[0] > target_h or 
                                       ridge_costmap.shape[1] > target_w) else cv2.INTER_LINEAR
    resized = cv2.resize(ridge_costmap, (target_w, target_h), interpolation=interpolation)
    
    return resized

