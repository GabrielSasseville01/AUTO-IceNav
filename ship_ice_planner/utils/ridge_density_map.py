"""
Ridge density map generation from satellite imagery.

This module implements methods to generate ridge density maps from:
1. Optical satellite imagery (adapted from Lensu & Similä 2022 approach)
2. Ice concentration grids (inferred from compression patterns)

Based on:
- Lensu, M. & Similä, M. (2022). "Ice Ridge Density Signatures in High-Resolution SAR Images"
- Gegiuc et al. (2018). "Estimation of Degree of Sea Ice Ridging Based on Dual-Polarized C-Band SAR Data"
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage
from skimage import filters, morphology, feature
import cv2


def compute_ridge_density_from_optical_image(
    image: np.ndarray,
    ice_mask: Optional[np.ndarray] = None,
    window_size: int = 25,
    method: str = 'texture_variance',
    auto_scale_window: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute ridge density map from optical satellite imagery.
    
    Adapts the Lensu & Similä (2022) density-based approach for optical imagery.
    In optical images, ridges appear as:
    - Texture variations (rough vs smooth ice)
    - Edge density (many edges = ridging zones)
    - Shadow patterns (height variations)
    
    Args:
        image: Input image (grayscale or RGB). If RGB, will be converted to grayscale.
              Can be uint8 (0-255) or float (0-1). Higher bit depth images will be normalized.
        ice_mask: Binary mask where 1 = ice, 0 = water. If None, will be estimated.
        window_size: Size of local window for density computation (pixels)
        method: Method to use:
            - 'texture_variance': Local variance in texture (good for optical)
            - 'edge_density': Density of edges (ridges create many edges)
            - 'gradient_magnitude': Magnitude of gradients (height variations)
            - 'combined': Combination of all methods
        auto_scale_window: If True, scale window_size based on image resolution
                          (larger images get proportionally larger windows)
    
    Returns:
        Ridge density map (0-1, where 1 = maximum ridging)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to 0-1 range
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    elif gray.dtype == np.uint16:
        gray = gray.astype(np.float32) / 65535.0
    elif gray.max() > 1.0:
        # Assume it's in some other range, normalize to 0-1
        gray = gray.astype(np.float32)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    else:
        gray = gray.astype(np.float32)
    
    # Auto-scale window size based on image resolution
    # For high-res images (like TIF), use proportionally larger windows
    original_window_size = window_size
    if auto_scale_window:
        # Base resolution: assume 1000x1000 is "standard"
        base_resolution = 1000.0
        image_resolution = np.sqrt(gray.shape[0] * gray.shape[1])
        scale_factor = image_resolution / base_resolution
        window_size = int(window_size * scale_factor)
        # Keep window size reasonable (min 15, max 100)
        window_size = max(15, min(100, window_size))
        if verbose:
            print(f"  Auto-scaled window size: {original_window_size} → {window_size} (scale factor: {scale_factor:.2f})")
    
    if verbose:
        print(f"  Processing image: {gray.shape[1]}×{gray.shape[0]} pixels")
        print(f"  Window size: {window_size}×{window_size} pixels")
        print(f"  Method: {method}")
    
    # Estimate ice mask if not provided
    if ice_mask is None:
        if verbose:
            print("  Computing ice/water mask...")
        # Simple thresholding (can be improved with watershed segmentation)
        _, ice_mask = cv2.threshold((gray * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ice_mask = ice_mask.astype(bool)
        if verbose:
            ice_pixels = np.sum(ice_mask)
            total_pixels = ice_mask.size
            print(f"  Ice mask: {ice_pixels}/{total_pixels} pixels ({100*ice_pixels/total_pixels:.1f}% ice)")
    
    # Initialize density map
    density_map = np.zeros_like(gray)
    
    if method == 'texture_variance' or method == 'combined':
        if verbose:
            print("  Computing texture variance (this may take a while for large images)...")
        # Method 1: Local texture variance
        # Ridges create rough texture (high variance), smooth ice has low variance
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        if verbose:
            print(f"    Computing local mean (convolution with {window_size}×{window_size} kernel)...")
        local_mean = ndimage.convolve(gray, kernel, mode='reflect')
        if verbose:
            print(f"    Computing local variance...")
        local_variance = ndimage.convolve((gray - local_mean) ** 2, kernel, mode='reflect')
        
        # Normalize variance to 0-1 range
        variance_normalized = np.clip(local_variance / (local_variance.max() + 1e-6), 0, 1)
        if verbose:
            print(f"    Texture variance: min={variance_normalized.min():.4f}, max={variance_normalized.max():.4f}, mean={variance_normalized.mean():.4f}")
        
        if method == 'texture_variance':
            density_map = variance_normalized
        else:
            density_map += variance_normalized * 0.4
    
    if method == 'edge_density' or method == 'combined':
        if verbose:
            print("  Computing edge density...")
        # Method 2: Edge density (similar to bright-pixel percentage in SAR)
        # Ridges create many edges, smooth ice has fewer edges
        if verbose:
            print(f"    Running Canny edge detection...")
        edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        if verbose:
            edge_pixels = np.sum(edges)
            print(f"    Found {edge_pixels} edge pixels ({100*edge_pixels/edges.size:.2f}% of image)")
        
        # Compute local density of edges
        if verbose:
            print(f"    Computing edge density (convolution with {window_size}×{window_size} kernel)...")
        kernel = np.ones((window_size, window_size))
        edge_density = ndimage.convolve(edges.astype(float), kernel, mode='reflect')
        edge_density = edge_density / (window_size ** 2)  # Normalize to 0-1
        if verbose:
            print(f"    Edge density: min={edge_density.min():.4f}, max={edge_density.max():.4f}, mean={edge_density.mean():.4f}")
        
        if method == 'edge_density':
            density_map = edge_density
        else:
            density_map += edge_density * 0.3
    
    if method == 'gradient_magnitude' or method == 'combined':
        if verbose:
            print("  Computing gradient magnitude...")
        # Method 3: Gradient magnitude (height variations indicate ridges)
        if verbose:
            print(f"    Computing image gradients...")
        grad_y, grad_x = np.gradient(gray)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Compute local mean of gradient magnitude
        if verbose:
            print(f"    Computing local gradient density (convolution with {window_size}×{window_size} kernel)...")
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_grad_density = ndimage.convolve(gradient_magnitude, kernel, mode='reflect')
        
        # Normalize
        local_grad_density = np.clip(local_grad_density / (local_grad_density.max() + 1e-6), 0, 1)
        if verbose:
            print(f"    Gradient density: min={local_grad_density.min():.4f}, max={local_grad_density.max():.4f}, mean={local_grad_density.mean():.4f}")
        
        if method == 'gradient_magnitude':
            density_map = local_grad_density
        else:
            density_map += local_grad_density * 0.3
    
    # Apply ice mask (set water areas to 0)
    if verbose:
        print("  Applying ice mask and normalizing...")
    density_map[~ice_mask] = 0.0
    
    # Normalize final map to 0-1
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    if verbose:
        print(f"  Final density map: min={density_map.min():.4f}, max={density_map.max():.4f}, mean={density_map.mean():.4f}")
        print("  ✓ Ridge density computation complete!")
    
    return density_map


def compute_ridge_density_from_concentration_grid(
    concentration: np.ndarray,
    floe_edges: Optional[np.ndarray] = None,
    window_size: int = 25,
    compression_threshold: float = 0.7
) -> np.ndarray:
    """
    Compute ridge density map from ice concentration grid.
    
    Infers ridging from:
    - High concentration + compression patterns
    - Floe edge density (many edges = potential ridging zones)
    - Concentration gradients (compression creates gradients)
    
    Args:
        concentration: Ice concentration grid (0-1, where 1 = 100% ice)
        floe_edges: Binary mask of floe edges (optional, will be computed if None)
        window_size: Size of local window for density computation (pixels)
        compression_threshold: Concentration threshold for compression detection
    
    Returns:
        Ridge density map (0-1, where 1 = maximum ridging)
    """
    density_map = np.zeros_like(concentration)
    
    # Method 1: High concentration + compression
    # Ridges form where ice is compressed (high concentration + gradients)
    high_conc_mask = concentration >= compression_threshold
    
    # Compute concentration gradients (compression creates strong gradients)
    grad_y, grad_x = np.gradient(concentration)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Local mean of gradient in high-concentration areas
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_grad = ndimage.convolve(gradient_magnitude, kernel, mode='reflect')
    
    # Combine: high concentration AND high gradients = ridging
    compression_density = (concentration * local_grad) * high_conc_mask.astype(float)
    
    # Method 2: Floe edge density (if available)
    if floe_edges is not None:
        # Many edges indicate potential ridging zones
        edge_density = ndimage.convolve(floe_edges.astype(float), kernel, mode='reflect')
        edge_density = edge_density / (window_size ** 2)  # Normalize
        
        # Combine with compression density
        density_map = 0.6 * compression_density + 0.4 * edge_density
    else:
        density_map = compression_density
    
    # Normalize to 0-1
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    return density_map


def compute_ridge_density_from_segmented_image(
    segmented_image: np.ndarray,
    window_size: int = 25
) -> np.ndarray:
    """
    Compute ridge density from watershed-segmented image.
    
    Uses the segmented image (from watershed.py) to identify:
    - Edge density (many edges = ridging)
    - Floe size distribution (small floes = more ridging)
    
    Args:
        segmented_image: Output from watershed segmentation (markers image)
        window_size: Size of local window for density computation (pixels)
    
    Returns:
        Ridge density map (0-1, where 1 = maximum ridging)
    """
    # Extract edges (boundaries between floes)
    # In watershed output, edges are marked as -1
    edges = (segmented_image == -1).astype(float)
    
    # Compute local edge density (similar to bright-pixel percentage)
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    edge_density = ndimage.convolve(edges, kernel, mode='reflect')
    
    # Also consider floe size: many small floes indicate ridging
    # Count number of unique floe IDs in each window
    unique_floes = np.zeros_like(segmented_image, dtype=float)
    for i in range(segmented_image.shape[0] - window_size):
        for j in range(segmented_image.shape[1] - window_size):
            window = segmented_image[i:i+window_size, j:j+window_size]
            # Count unique floe IDs (excluding edges and background)
            unique_ids = np.unique(window[(window > 0) & (window != -1)])
            unique_floes[i+window_size//2, j+window_size//2] = len(unique_ids)
    
    # Normalize floe count density
    if unique_floes.max() > 0:
        unique_floes = unique_floes / unique_floes.max()
    
    # Combine edge density and floe count density
    density_map = 0.7 * edge_density + 0.3 * unique_floes
    
    # Normalize to 0-1
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    return density_map


def ridge_density_to_ridge_properties(
    density_map: np.ndarray,
    position_map: Optional[np.ndarray] = None,
    base_thickness: float = 0.5,
    max_thickness: float = 5.0,
    base_resistance: float = 5000.0,
    max_resistance: float = 50000.0
) -> Dict[str, Any]:
    """
    Convert ridge density map to ridge properties for simulation.
    
    Maps density (0-1) to physical ridge properties:
    - Thickness (sail height)
    - Packing density
    - Resistance coefficient
    
    Args:
        density_map: Ridge density map (0-1)
        position_map: Optional map of (x, y) positions for each pixel
        base_thickness: Minimum ridge thickness (m)
        max_thickness: Maximum ridge thickness (m)
        base_resistance: Base resistance coefficient (N·s²/m⁴)
        max_resistance: Maximum resistance coefficient (N·s²/m⁴)
    
    Returns:
        Dictionary with:
            - 'thickness_map': Map of ridge thicknesses (m)
            - 'packing_density_map': Map of packing densities (0-1)
            - 'resistance_map': Map of resistance coefficients
            - 'ridge_zones': List of ridge zone dictionaries (position, width, length, properties)
    """
    # Map density to thickness
    thickness_map = base_thickness + (max_thickness - base_thickness) * density_map
    
    # Map density to packing density (higher density = more consolidated)
    packing_density_map = 0.6 + 0.3 * density_map  # 0.6 to 0.9
    
    # Map density to resistance
    resistance_map = base_resistance + (max_resistance - base_resistance) * density_map
    
    # Extract ridge zones (regions with density above threshold)
    ridge_threshold = 0.3  # Minimum density to be considered a ridge
    ridge_mask = density_map >= ridge_threshold
    
    # Find connected components (individual ridge zones)
    from skimage import measure
    labeled_ridges = measure.label(ridge_mask, connectivity=2)
    
    ridge_zones = []
    for ridge_id in range(1, labeled_ridges.max() + 1):
        ridge_pixels = (labeled_ridges == ridge_id)
        
        # Compute properties of this ridge zone
        zone_density = density_map[ridge_pixels].mean()
        zone_thickness = thickness_map[ridge_pixels].mean()
        zone_packing = packing_density_map[ridge_pixels].mean()
        zone_resistance = resistance_map[ridge_pixels].mean()
        
        # Find bounding box
        y_indices, x_indices = np.where(ridge_pixels)
        if len(y_indices) == 0:
            continue
        
        min_y, max_y = y_indices.min(), y_indices.max()
        min_x, max_x = x_indices.min(), x_indices.max()
        
        width = (max_x - min_x) * 1.2  # Add margin
        length = (max_y - min_y) * 1.2
        
        # Get center position
        center_y = (min_y + max_y) / 2
        center_x = (min_x + max_x) / 2
        
        if position_map is not None:
            center_pos = position_map[int(center_y), int(center_x)]
        else:
            center_pos = (center_x, center_y)  # Pixel coordinates
        
        ridge_zones.append({
            'position': center_pos,
            'width': width,
            'length': length,
            'thickness': zone_thickness,
            'packing_density': zone_packing,
            'resistance_coefficient': zone_resistance,
            'density': zone_density
        })
    
    return {
        'thickness_map': thickness_map,
        'packing_density_map': packing_density_map,
        'resistance_map': resistance_map,
        'ridge_zones': ridge_zones,
        'density_map': density_map
    }


def load_and_compute_ridge_density(
    image_path: str,
    method: str = 'auto',
    window_size: int = 25
) -> Dict[str, Any]:
    """
    Load image and compute ridge density map.
    
    Convenience function that handles loading and processing.
    
    Args:
        image_path: Path to image file (PNG, JPG, etc.)
        method: Method to use ('auto', 'optical', 'segmented', or 'concentration')
        window_size: Size of local window for density computation
    
    Returns:
        Dictionary with density map and properties
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if method == 'auto':
        # Try to detect method based on image characteristics
        # For now, default to optical
        method = 'optical'
    
    if method == 'optical':
        density_map = compute_ridge_density_from_optical_image(
            image, window_size=window_size, method='combined'
        )
    elif method == 'segmented':
        # Would need segmented image - for now, use optical
        from ship_ice_planner.image_process.watershed import segment_ice
        segmented = segment_ice(image)
        density_map = compute_ridge_density_from_segmented_image(
            segmented, window_size=window_size
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to properties
    properties = ridge_density_to_ridge_properties(density_map)
    properties['density_map'] = density_map
    
    return properties

