"""
Generate ridge density costmap from satellite imagery for path planning.

Similar to generate_poly_map.py, but creates a coarse ridge density costmap
that can be used in the simulation to avoid ridges during path planning.
"""

import cv2
import numpy as np
import pickle
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import ridge_density_map
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ship_ice_planner.utils.ridge_density_map import (
    compute_ridge_density_from_optical_image
)


def downsample_costmap(costmap: np.ndarray, target_scale: float, original_scale: float) -> np.ndarray:
    """
    Resample a costmap to match target scale, preserving physical dimensions.
    
    Args:
        costmap: Original costmap (H, W)
        target_scale: Target scale (meters per pixel) for simulation
        original_scale: Original scale (meters per pixel) of input image
    
    Returns:
        Resampled costmap (may be upsampled or downsampled depending on scales)
    """
    scale_factor = original_scale / target_scale
    
    if abs(scale_factor - 1.0) < 1e-6:
        # Same scale, return as is
        return costmap
    
    # Compute target dimensions to preserve physical area
    # If target_scale < original_scale (e.g., 0.5 < 1.0), we need MORE pixels (upsample)
    # If target_scale > original_scale (e.g., 2.0 > 1.0), we need FEWER pixels (downsample)
    # Physical dimensions: original_pixels * original_scale = target_pixels * target_scale
    # So: target_pixels = original_pixels * (original_scale / target_scale)
    target_h = int(costmap.shape[0] * scale_factor)
    target_w = int(costmap.shape[1] * scale_factor)
    
    # Use area-based resampling
    from scipy.ndimage import zoom
    resampled = zoom(costmap, (target_h / costmap.shape[0], target_w / costmap.shape[1]), order=1)
    
    return resampled


def generate_ridge_costmap(
    input_image_path: str,
    satellite_image_path: str = None,
    crop_size: int = 600,
    costmap_scale: float = 1.0,
    method: str = 'combined',
    window_size: int = 25,
    output_path: str = None,
    visualize: bool = False
):
    """
    Generate a ridge density costmap from satellite imagery.
    
    Args:
        input_image_path: Path to segmented PNG image (used for cropping/extent)
        satellite_image_path: Optional path to original satellite TIF image (better for ridge detection)
        crop_size: Size to crop image to (square crop from top-left, pixels)
        costmap_scale: Scale for output costmap (meters per pixel). Lower = coarser
        method: Method for ridge density computation ('texture_variance', 'edge_density', 'gradient_magnitude', 'combined')
        window_size: Window size for ridge density computation
        output_path: Output path for pickle file
        visualize: Whether to show visualization
    """
    
    # Determine pixel scale of input image (assume 1m per pixel for segmented image)
    # This should match the scale used in generate_poly_map
    input_pixel_scale = 1.0  # meters per pixel for segmented image
    
    # Load segmented image first to get its dimensions
    print(f"Loading segmented image: {input_image_path}")
    segmented_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if segmented_image is None:
        raise FileNotFoundError(f"Could not load image: {input_image_path}")
    segmented_h, segmented_w = segmented_image.shape[:2]
    print(f"  Segmented image shape: {segmented_image.shape}")
    
    # Use satellite image if provided (better quality for ridge detection)
    if satellite_image_path and Path(satellite_image_path).exists():
        print(f"Loading satellite image: {satellite_image_path}")
        # Load TIF (may be grayscale or have different bit depth)
        satellite_image = cv2.imread(satellite_image_path, cv2.IMREAD_UNCHANGED)
        if satellite_image is None:
            raise ValueError(f"Could not load TIF image: {satellite_image_path}")
        print(f"  Loaded TIF: shape={satellite_image.shape}, dtype={satellite_image.dtype}, "
              f"min={satellite_image.min()}, max={satellite_image.max()}")
        
        # Calculate scale factor between satellite and segmented images
        # This tells us how many satellite pixels correspond to one segmented pixel
        scale_h = satellite_image.shape[0] / segmented_h
        scale_w = satellite_image.shape[1] / segmented_w
        print(f"  Scale factors: {scale_h:.2f}x (height), {scale_w:.2f}x (width)")
        
        # Crop satellite image to match the same physical region as the segmented image crop
        # The segmented image is cropped to [:crop_size, :crop_size]
        # So we need to crop the satellite image proportionally
        satellite_crop_h = int(crop_size * scale_h)
        satellite_crop_w = int(crop_size * scale_w)
        print(f"  Cropping satellite image to {satellite_crop_h}x{satellite_crop_w} pixels "
              f"(equivalent to {crop_size}x{crop_size} pixels in segmented image)")
        
        satellite_image = satellite_image[:satellite_crop_h, :satellite_crop_w]
        
        # Compute ridge density from satellite image
        print(f"Computing ridge density map from satellite image...")
        ridge_density = compute_ridge_density_from_optical_image(
            satellite_image,
            window_size=window_size,
            method=method,
            auto_scale_window=True,
            verbose=True
        )
        
        # The satellite image pixel scale is different - it's higher resolution
        # If segmented image is 1 m/pixel, then satellite is 1/scale_h m/pixel
        satellite_pixel_scale = input_pixel_scale / scale_h  # More pixels per meter (higher resolution)
        
    else:
        # Fall back to segmented image
        if satellite_image_path:
            print(f"Warning: Satellite image not found at {satellite_image_path}, using segmented image")
        
        print(f"Loading segmented image: {input_image_path}")
        image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {input_image_path}")
        
        # Crop to match crop_size
        image = image[:crop_size, :crop_size, :]
        
        # Compute ridge density from segmented image (less accurate, but works)
        print(f"Computing ridge density map from segmented image...")
        ridge_density = compute_ridge_density_from_optical_image(
            image,
            window_size=window_size,
            method=method,
            auto_scale_window=False,  # Already cropped to reasonable size
            verbose=True
        )
        
        satellite_pixel_scale = input_pixel_scale
    
    print(f"\nRidge density map shape: {ridge_density.shape}")
    print(f"Ridge density range: [{ridge_density.min():.4f}, {ridge_density.max():.4f}]")
    print(f"Ridge density mean: {ridge_density.mean():.4f}")
    
    # Downsample to coarser resolution for simulation
    print(f"\nDownsampling to costmap scale: {costmap_scale} m/pixel (from {satellite_pixel_scale} m/pixel)")
    ridge_costmap = downsample_costmap(ridge_density, costmap_scale, satellite_pixel_scale)
    
    print(f"Downsampled costmap shape: {ridge_costmap.shape}")
    print(f"Downsampled density range: [{ridge_costmap.min():.4f}, {ridge_costmap.max():.4f}]")
    
    # Compute dimensions in meters
    width_m = ridge_costmap.shape[1] * costmap_scale
    length_m = ridge_costmap.shape[0] * costmap_scale
    
    # Prepare output data
    output_data = {
        'ridge_costmap': ridge_costmap,  # 2D array, values 0-1 (ridge density)
        'scale': costmap_scale,  # meters per pixel
        'width': width_m,  # width in meters
        'length': length_m,  # length in meters
        'shape': ridge_costmap.shape,  # (height, width) in pixels
        'metadata': {
            'input_image': input_image_path,
            'satellite_image': satellite_image_path,
            'crop_size': crop_size,
            'method': method,
            'window_size': window_size,
            'original_shape': ridge_density.shape,
            'original_scale': satellite_pixel_scale,
            'downsample_factor': satellite_pixel_scale / costmap_scale
        }
    }
    
    # Save to pickle file
    if output_path is None:
        input_stem = Path(input_image_path).stem
        output_path = f"data/ridge_costmap_{input_stem}.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Saved ridge costmap to: {output_path}")
    print(f"  Dimensions: {width_m:.1f} m × {length_m:.1f} m")
    print(f"  Grid size: {ridge_costmap.shape[1]} × {ridge_costmap.shape[0]} pixels")
    print(f"  Scale: {costmap_scale} m/pixel")
    
    # Visualization
    if visualize:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original density map
        im1 = axes[0].imshow(ridge_density, cmap='hot', origin='upper', vmin=0, vmax=1)
        axes[0].set_title(f'Original Ridge Density Map\n{ridge_density.shape[1]}×{ridge_density.shape[0]} pixels')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0], label='Ridge Density')
        
        # Downsampled costmap
        im2 = axes[1].imshow(ridge_costmap, cmap='hot', origin='upper', vmin=0, vmax=1)
        axes[1].set_title(f'Downsampled Ridge Costmap\n{ridge_costmap.shape[1]}×{ridge_costmap.shape[0]} pixels\n{costmap_scale} m/pixel')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[1], label='Ridge Density')
        
        plt.tight_layout()
        
        # Save visualization
        vis_path = output_path.replace('.pkl', '_visualization.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: {vis_path}")
        
        try:
            plt.show()
        except:
            pass  # Ignore display errors in headless environments
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate ridge density costmap from satellite imagery for path planning'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/MEDEA_fram_20100629.png',
        help='Input segmented image path (used for extent/cropping)'
    )
    parser.add_argument(
        '--satellite',
        type=str,
        default=None,
        help='Optional: Original satellite TIF image path (better for ridge detection). '
             'If not provided, uses segmented image.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle file path (default: data/ridge_costmap_<input_stem>.pkl)'
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=600,
        help='Size to crop image to (square crop from top-left, pixels)'
    )
    parser.add_argument(
        '--costmap_scale',
        type=float,
        default=1.0,
        help='Scale for output costmap (meters per pixel). Lower = coarser resolution. Default: 1.0'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='combined',
        choices=['texture_variance', 'edge_density', 'gradient_magnitude', 'combined'],
        help='Method for ridge density computation'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=25,
        help='Window size for ridge density computation (pixels)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of ridge density maps'
    )
    
    args = parser.parse_args()
    
    generate_ridge_costmap(
        input_image_path=args.input,
        satellite_image_path=args.satellite,
        crop_size=args.crop_size,
        costmap_scale=args.costmap_scale,
        method=args.method,
        window_size=args.window_size,
        output_path=args.output,
        visualize=args.visualize
    )


if __name__ == '__main__':
    main()

