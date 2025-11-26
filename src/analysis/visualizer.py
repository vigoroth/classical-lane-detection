"""
Visualization utilities for analysis.

Functions for creating comparison grids, pipeline visualizations, and plots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import sys

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


def create_comparison_grid(original_images: List, processed_images: List,
                          titles: List[str], output_path: str):
    """
    Create side-by-side comparison grid.

    Args:
        original_images: List of original images (BGR format)
        processed_images: List of processed images (BGR format)
        titles: List of image titles
        output_path: Where to save the grid

    Returns:
        None (saves to file)
    """
    n = len(original_images)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        if idx < n:
            # Stack original and processed vertically
            orig_rgb = cv2.cvtColor(original_images[idx], cv2.COLOR_BGR2RGB)
            proc_rgb = cv2.cvtColor(processed_images[idx], cv2.COLOR_BGR2RGB)
            combined = np.vstack([orig_rgb, proc_rgb])

            ax.imshow(combined)
            ax.set_title(titles[idx], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {output_path}")


def plot_intermediate_steps(image_path: str, config: dict, output_dir: str):
    """
    Show all 9 pipeline steps for debugging.

    Args:
        image_path: Path to input image
        config: Configuration dictionary
        output_dir: Directory to save visualization

    Returns:
        None (saves to file)
    """
    from preprocessing import grayscale, apply_gaussian_blur, define_roi
    from edge_detection import detect_edge_canny
    from line_detection import detect_line_hough, separate_lanes, average_lane_line, draw_lane_lines

    # Load and process
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    gray = grayscale(img)
    blurred = apply_gaussian_blur(gray, config['preprocessing']['blur_kernel'])

    # ROI
    h, w = img.shape[:2]
    roi_verts = config['roi']['vertices']
    if roi_verts is None:
        roi_verts = np.array([[
            (int(w * 0.1), h),
            (int(w * 0.45), int(h * 0.6)),
            (int(w * 0.55), int(h * 0.6)),
            (int(w * 0.9), h)
        ]], dtype=np.int32)
    else:
        roi_verts = np.array([roi_verts], dtype=np.int32)

    masked = define_roi(blurred, roi_verts)
    edges = detect_edge_canny(masked, config['edge_detection']['canny_low'],
                             config['edge_detection']['canny_high'])

    lines = detect_line_hough(edges, **config['hough_transform'])
    left_lines, right_lines = separate_lanes(lines, img.shape)
    left_lane = average_lane_line(left_lines, h)
    right_lane = average_lane_line(right_lines, h)
    final = draw_lane_lines(img.copy(), left_lane, right_lane)

    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        gray,
        blurred,
        masked,
        edges,
        edges,  # For Hough viz placeholder
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    ]
    titles = ['1. Original', '2. Grayscale', '3. Blurred', '4. ROI Masked',
              '5. Canny Edges', '6. Detected Lines', '7. Separated Lanes',
              '8. Averaged Lanes', '9. Final Output']

    for ax, img_data, title in zip(axes, images, titles):
        if len(img_data.shape) == 2:
            ax.imshow(img_data, cmap='gray')
        else:
            ax.imshow(img_data)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    output_name = Path(image_path).stem + '_pipeline_steps.png'
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Pipeline steps saved to: {output_path}")


def plot_detection_rate_chart(metrics: Dict, output_path: str):
    """
    Create bar chart showing detection rates.

    Args:
        metrics: Dictionary from compute_detection_rate()
        output_path: Where to save the chart

    Returns:
        None (saves to file)
    """
    categories = ['Left Only', 'Right Only', 'Both Lanes', 'At Least One']
    values = [
        metrics['left_lane_detected'],
        metrics['right_lane_detected'],
        metrics['both_lanes_detected'],
        metrics['at_least_one_detected']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
    ax.set_ylabel('Number of Images')
    ax.set_title(f'Lane Detection Results (Total: {metrics["total_images"]} images)')
    ax.set_ylim(0, metrics['total_images'] + 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Detection rate chart saved to: {output_path}")


def plot_failure_modes(failure_modes: Dict, output_path: str):
    """
    Create pie chart showing failure mode distribution.

    Args:
        failure_modes: Dictionary from analyze_failure_modes()
        output_path: Where to save the chart

    Returns:
        None (saves to file)
    """
    labels = []
    sizes = []
    colors = []

    mode_config = {
        'success': ('Success', '#2ecc71'),
        'only_left_detected': ('Only Left', '#3498db'),
        'only_right_detected': ('Only Right', '#e74c3c'),
        'no_lines_detected': ('No Lines', '#95a5a6')
    }

    for mode, (label, color) in mode_config.items():
        count = len(failure_modes[mode])
        if count > 0:
            labels.append(f'{label} ({count})')
            sizes.append(count)
            colors.append(color)

    if not sizes:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Lane Detection Failure Mode Distribution')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Failure modes chart saved to: {output_path}")


def plot_processing_times(results: Dict, output_path: str):
    """
    Create histogram of processing times.

    Args:
        results: Dictionary from batch_process_images()
        output_path: Where to save the chart

    Returns:
        None (saves to file)
    """
    times = [r['processing_time'] for r in results.values()]

    if not times:
        print("No timing data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(times, bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel('Processing Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Processing Times')
    ax.axvline(np.mean(times), color='red', linestyle='dashed', linewidth=2,
               label=f'Mean: {np.mean(times):.2f} ms')
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Processing times histogram saved to: {output_path}")
