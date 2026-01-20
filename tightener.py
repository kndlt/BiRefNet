"""
Purpose:

Input is a image that has just been background matted.
But the matted image contains white seams around the edges.
Or it could be black/red/magenta seams around the deges.
I want to remove those seams.

As an input, we provide:
1. Seam's width in pixels.

Then the code will change the RGBs of those seam pixels to match the nearby non-seam pixels.

RUN:
python tightener.py -i sample_nobg.png


"""

import argparse
import numpy as np
from PIL import Image
import cv2
import sys
import os


def detect_edge_mask(alpha_channel, seam_width):
    """
    Detect edge pixels near the alpha channel boundary.
    Returns a mask where seam pixels are marked.
    """
    # Convert alpha to numpy array
    alpha_np = np.array(alpha_channel).astype(np.float32) / 255.0
    
    # Find semi-transparent edge pixels (alpha between 0.05 and 0.95)
    edge_mask = ((alpha_np > 0.05) & (alpha_np < 0.95)).astype(np.uint8) * 255
    
    # Create binary mask for FULLY opaque pixels only (alpha = 255)
    _, binary_mask = cv2.threshold((alpha_np * 255).astype(np.uint8), 254, 255, cv2.THRESH_BINARY)
    
    # Get inner safe region - erode by seam_width pixels
    kernel = np.ones((3, 3), np.uint8)
    inner_region = cv2.erode(binary_mask, kernel, iterations=seam_width)
    
    # Seam is everything between the inner region and the outer edge
    # This includes semi-transparent pixels + opaque pixels within seam_width of edge
    seam_mask = cv2.bitwise_or(edge_mask, binary_mask - inner_region)
    
    return seam_mask, inner_region, alpha_np


def inpaint_seams(image_rgb, alpha_np, seam_mask, inner_region, seam_width):
    """
    Replace seam pixels with weighted average of nearby inner pixels.
    Weights are based on inverse distance.
    """
    # Convert to numpy arrays
    img_np = np.array(image_rgb).astype(np.float32)
    
    # Get coordinates of seam pixels and inner pixels
    seam_coords = np.argwhere(seam_mask > 0)
    inner_coords = np.argwhere(inner_region > 0)
    
    if len(inner_coords) == 0:
        print("Warning: No inner region found, returning original")
        return image_rgb, (alpha_np * 255).astype(np.uint8)
    
    if len(seam_coords) == 0:
        return image_rgb, (alpha_np * 255).astype(np.uint8)
    
    result = img_np.copy()
    alpha_result = alpha_np.copy()
    
    # For each seam pixel, find K nearest inner pixels
    from scipy.spatial import cKDTree
    tree = cKDTree(inner_coords)
    
    # Search within 1 pixel radius only
    search_radius = 1.5  # Slightly more than 1 to catch diagonal neighbors
    
    # Weight colors by inverse distance
    for seam_px in seam_coords:
        # Find all inner pixels within radius
        nearby_indices = tree.query_ball_point(seam_px, r=search_radius)
        
        if len(nearby_indices) == 0:
            # If no pixels within 1 pixel, fall back to nearest pixel
            dist, idx = tree.query(seam_px, k=1)
            nearby_indices = [idx]
        
        # Get coordinates and calculate distances
        nearby_coords = inner_coords[nearby_indices]
        dists = np.linalg.norm(nearby_coords - seam_px, axis=1)
        
        # Handle case where distance is 0 (shouldn't happen but just in case)
        dists = np.where(dists < 1e-6, 1e-6, dists)
        
        # Inverse distance weighting
        weights = 1.0 / dists
        weights = weights / weights.sum()
        
        # Weighted average of RGB values
        weighted_color = np.zeros(3, dtype=np.float32)
        for w, idx in zip(weights, nearby_indices):
            inner_px = inner_coords[idx]
            weighted_color += w * img_np[inner_px[0], inner_px[1]]
        
        result[seam_px[0], seam_px[1]] = weighted_color
        
        # Make edge pixels more opaque to reduce halo
        alpha_result[seam_px[0], seam_px[1]] = max(alpha_result[seam_px[0], seam_px[1]], 0.98)
    
    return Image.fromarray(result.astype(np.uint8)), (alpha_result * 255).astype(np.uint8)


def remove_color_seams(image_path, seam_width=2, threshold=None, output_path=None):
    """
    Remove color seams from a background-matted image.
    
    Args:
        image_path: Path to input image with alpha channel
        seam_width: Width of seam in pixels to remove
        threshold: Optional alpha threshold value (0-255). Pixels above this become opaque, below become transparent.
        output_path: Path to save output (optional)
    
    Returns:
        PIL Image with seams removed
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    
    # Check if image has alpha channel
    if img.mode != 'RGBA':
        print(f"Warning: Image doesn't have alpha channel (mode: {img.mode})")
        if img.mode == 'RGB':
            print("Adding full alpha channel...")
            img.putalpha(255)
        else:
            img = img.convert('RGBA')
    
    # Split into RGB and alpha
    rgb = img.convert('RGB')
    alpha = img.getchannel('A')
    
    # Apply threshold if specified
    if threshold is not None:
        print(f"Applying alpha threshold at {threshold}...")
        alpha_np = np.array(alpha)
        alpha_np = np.where(alpha_np >= threshold, 255, 0).astype(np.uint8)
        alpha = Image.fromarray(alpha_np)
        print(f"✓ Alpha thresholded")
    
    print(f"Detecting seams with width: {seam_width} pixels...")
    
    # Detect seam mask
    seam_mask, inner_region, alpha_np = detect_edge_mask(alpha, seam_width)
    
    num_seam_pixels = np.sum(seam_mask > 0)
    print(f"Found {num_seam_pixels} seam pixels to fix")
    
    if num_seam_pixels == 0:
        print("No seams detected!")
        return img
    
    print(f"Replacing seam colors and adjusting alpha...")
    
    # Inpaint seams and get new alpha
    fixed_rgb, fixed_alpha = inpaint_seams(rgb, alpha_np, seam_mask, inner_region, seam_width)
    
    # Combine with fixed alpha
    result = fixed_rgb.copy()
    result.putalpha(Image.fromarray(fixed_alpha))
    
    # Save if output path provided
    if output_path:
        result.save(output_path)
        print(f"✓ Saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Remove color seams from background-matted images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tightener.py -i image_nobg.png
  python tightener.py -i image_nobg.png -w 3
  python tightener.py -i image_nobg.png -t 128 -w 2
  python tightener.py -i image_nobg.png -o fixed.png -t 200 -w 2
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input image with alpha channel'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save output image (default: input_name_fixed.png)'
    )
    
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=1,
        help='Seam width in pixels to remove (default: 1)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=200,
        help='Alpha threshold value (0-255). Pixels >= threshold become opaque, < threshold become transparent. Applied before seam removal (default: 200).'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        output_path = f"{base_name}_fixed.png"
    else:
        output_path = args.output
    
    # Process image
    try:
        result = remove_color_seams(args.input, seam_width=args.width, threshold=args.threshold, output_path=output_path)
        print(f"✓ Seam removal complete!")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

