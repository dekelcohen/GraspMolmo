# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:20:16 2025

@author: DEKELCO (modified)

This script annotates an image with a bounding box and pixel rulers.
New: supports an explicit --normalized flag which interprets the bbox values
as normalized coordinates. You can choose normalization range:
- 0..1   (typical ML convention)
- 0..1000 (some labeling tools convention)

Usage examples:
    # absolute pixel coords (default)
    python annotate_image.py --image "image.png" --bbox "14,38,126,90"

    # normalized coords (0..1)
    python annotate_image.py --image "image.png" --bbox "0.02,0.05,0.25,0.18" --normalized 1

    # normalized coords (0..1000)
    # Note: VLMs may extract y,x format (instead of x,y) - depending on their training and prompt
    python "D:\Docs\test6\Projects\Vision\Object Detection\annotate_image\annotate_image.py" --image "D:\Docs\test6\Projects\Vision\DroneDetector\data\web_images\small_flying_drone_bird_sky_3.PNG" --bbox "320,610,330,620" --normalized 1000
"""

import argparse
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def load_image(image_path):
    """Load image from file path or URL and return as numpy array."""
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        raise Exception(f"Failed to load image: {e}")


def scale_bbox(bbox_vals, img_shape, normalized=False, norm_range=1000.0):
    """Scale bbox values to pixel coordinates.

    Args:
        bbox_vals: iterable of 4 numbers (x1,y1,x2,y2) as floats or ints.
        img_shape: image.shape (h, w, ...)
        normalized: if False, interpret bbox_vals as absolute pixel coords;
                    if True, interpret as normalized values in 0..norm_range.
        norm_range: normalization range (1.0 or 1000.0).

    Returns:
        tuple of ints (x1, y1, x2, y2) in pixel coordinates.
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox_vals    

    if normalized:
        # Validate range
        for v in (x1, y1, x2, y2):
            if v < 0 or v > norm_range:
                raise ValueError(f"Normalized bbox value {v} out of range 0..{norm_range}")

        # Scale: normalized is assumed axis-aligned with pixel axes
        x1_px = int(round((x1 / norm_range) * w))
        x2_px = int(round((x2 / norm_range) * w))
        y1_px = int(round((y1 / norm_range) * h))
        y2_px = int(round((y2 / norm_range) * h))
    else:
        # Already pixel coordinates; just convert to int
        x1_px, y1_px, x2_px, y2_px = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        print('Treat BBox as absolute pixel coordinates. use --normalized to pass 0-1000 or 0-1 normalized coordinates')

    # Optional: clamp to image bounds
    x1_px = max(0, min(w, x1_px))
    x2_px = max(0, min(w, x2_px))
    y1_px = max(0, min(h, y1_px))
    y2_px = max(0, min(h, y2_px))

    # Validate order
    if x2_px <= x1_px or y2_px <= y1_px:
        raise ValueError(f"Invalid bbox after scaling to pixels: {(x1_px,y1_px,x2_px,y2_px)}")

    return (x1_px, y1_px, x2_px, y2_px)


def draw_bbox_and_rulers(image, bbox, output_path=None, ticks_every=20):
    """Draw bounding box and rulers on image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Display image
    ax.imshow(image)

    # Extract bbox
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Set axis limits to match image dimensions
    h, w = image.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Inverted because (0,0) is top-left in images

    # Add X ruler at bottom
    ax.set_xlabel('X (pixels)', fontsize=10)
    ax.xaxis.set_ticks(np.arange(0, w + 1, ticks_every))
    ax.xaxis.set_ticklabels(np.arange(0, w + 1, ticks_every))

    # Add Y ruler on left
    ax.set_ylabel('Y (pixels)', fontsize=10)
    ax.yaxis.set_ticks(np.arange(0, h + 1, ticks_every))
    ax.yaxis.set_ticklabels(np.arange(0, h + 1, ticks_every))

    # Ensure aspect ratio and no margins
    ax.set_aspect('equal')
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Optional: Remove tick labels if too dense
    ax.tick_params(axis='both', which='both', length=0)

    # Show or save
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved annotated image to {output_path}")
    else:
        plt.show()

    plt.close()


def parse_bbox_string(bbox_str):
    """Parse bbox string 'x1,y1,x2,y2' into tuple of floats."""
    parts = [p.strip() for p in bbox_str.split(',') if p.strip() != '']
    if len(parts) != 4:
        raise ValueError(f"Bbox must have 4 values: x1,y1,x2,y2. Got: {bbox_str}")
    try:
        vals = tuple(float(p) for p in parts)
    except Exception as e:
        raise ValueError(f"Failed to parse bbox numbers: {e}")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Annotate image with bbox and rulers.")
    parser.add_argument('--image', required=True, help='Path to image or URL')
    parser.add_argument('--bbox', required=True, help='Bounding box as x1,y1,x2,y2')
    parser.add_argument('--output', help='Output path to save annotated image')
    parser.add_argument(
        '--normalized',
        type=float,
        choices=[1.0, 1000.0],
        default=0.0,
        help='If set, interpret bbox as normalized coordinates. Choose range: 1.0 for 0..1, 1000.0 for 0..1000. Default is off (pixel coords).'
    )
    parser.add_argument('--ticks_every', type=int, default=20, help='Pixel spacing for ruler ticks')

    args = parser.parse_args()

    # Parse bbox
    bbox_vals = parse_bbox_string(args.bbox)

    # Load image
    try:
        image = load_image(args.image)
    except Exception as e:
        raise Exception(f"Error loading image: {e}")

    # Scale bbox if needed
    try:
        normalized_flag = args.normalized in (1.0, 100.0, 1000.0)
        bbox_px = scale_bbox(bbox_vals, image.shape, normalized=normalized_flag, norm_range=args.normalized if normalized_flag else 1000.0)
    except Exception as e:
        raise Exception(f"Error scaling bbox: {e}")

    # Draw and save/show
    draw_bbox_and_rulers(image, bbox_px, args.output, ticks_every=args.ticks_every)


if __name__ == "__main__":
    main()
