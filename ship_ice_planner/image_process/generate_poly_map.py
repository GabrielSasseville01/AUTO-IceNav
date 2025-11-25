import cv2
import numpy as np
import pickle
import argparse


def poly_area(vertices):
    """Calculate the area of a polygon using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def create_color_mask(image, color):
    """
    Create a binary mask for pixels matching the given color.
    Uses pure numpy for cross-platform consistency (Intel/ARM).
    """
    # Ensure color is uint8 for proper comparison
    color = np.array(color, dtype=np.uint8)
    # Compare all channels and combine - more reliable than cv2.inRange across platforms
    mask = np.all(image == color, axis=2).astype(np.uint8) * 255
    return mask


def main():
    parser = argparse.ArgumentParser(description='Convert segmented ice image to polygons')
    parser.add_argument('--input', type=str, default='data/MEDEA_fram_20100629.png',
                        help='Input segmented image path')
    parser.add_argument('--output', type=str, default='data/ice_floes_MEDEA_fram_20100629.pkl',
                        help='Output pickle file path')
    parser.add_argument('--crop_size', type=int, default=600,
                        help='Size to crop image to (square crop from top-left)')
    parser.add_argument('--min_area', type=float, default=50,
                        help='Minimum contour area to keep')
    parser.add_argument('--epsilon_factor', type=float, default=0.0005,
                        help='Epsilon factor for polygon approximation (smaller = more detailed)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of extracted polygons')
    
    args = parser.parse_args()

    # Load the image with explicit flags for consistent decoding
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")
    
    # Ensure uint8 dtype for consistent behavior
    image = image.astype(np.uint8)

    # Reshape to (H*W, 3) and find unique colors
    image = image[:args.crop_size, :args.crop_size, :]
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    # Filter out black (water) - assuming black is [0,0,0]
    unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]

    print(f"Found {len(unique_colors)} unique segment colors")

    # Create output image to draw polygons
    output = image.copy()

    # Store all polygons
    all_polygons = []

    for color in unique_colors:
        # Create binary mask for this color using platform-independent method
        mask = create_color_mask(image, color)
        
        # Find contours - use RETR_EXTERNAL for outer contours only
        # CHAIN_APPROX_NONE keeps all contour points (detailed, follows segmentation exactly)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            # Ensure contour is contiguous array with correct dtype for consistent behavior
            contour = np.ascontiguousarray(contour, dtype=np.int32)
            
            # Filter small contours by area
            area = cv2.contourArea(contour)
            if area < args.min_area:  # skip very small regions
                continue
            
            # Smooth contour slightly while keeping detail
            # epsilon controls approximation accuracy (smaller = more detailed)
            arc_length = cv2.arcLength(contour, True)
            # Use a minimum epsilon to avoid numerical precision issues
            epsilon = max(args.epsilon_factor * arc_length, 0.1)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Skip degenerate polygons (need at least 3 vertices)
            if len(approx) < 3:
                continue
                
            ice_flow_vertices = approx[:, 0, :].astype(np.float64)
            
            # Compute centre and radius
            centre = tuple(np.mean(ice_flow_vertices, axis=0))
            # Radius as max distance from centre to any vertex
            distances = np.linalg.norm(ice_flow_vertices - centre, axis=1)
            radius = np.max(distances)
            
            # Compute area using shoelace formula
            polygon_area = poly_area(ice_flow_vertices)
            
            all_polygons.append({
                'vertices': ice_flow_vertices,
                'area': polygon_area,
                'centre': centre,
                'radius': radius,
            })

    with open(args.output, "wb") as f:
        pickle.dump(all_polygons, f)
        f.close()

    print(f"Extracted {len(all_polygons)} polygons")
    print(f"Saved to {args.output}")

    if args.visualize:
        # Draw all polygons on output image
        # Use random colors for visualization
        for polygon_dict in all_polygons:
            # Convert back to int32 for drawing
            polygon = polygon_dict['vertices'].astype(np.int32)
            # Random color for each polygon
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(output, [polygon], -1, color, 2)

        # Show results
        cv2.imshow('Original Image', image)
        cv2.imshow('Extracted Polygons', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
