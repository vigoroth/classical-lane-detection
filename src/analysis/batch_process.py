"""
Batch processing script for all test images.

Processes multiple images and collects performance metrics.
"""

import cv2
import time
import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from main import load_config, detect_lanes
from preprocessing import grayscale, apply_gaussian_blur, define_roi
from edge_detection import detect_edge_canny
from line_detection import detect_line_hough, separate_lanes, average_lane_line


def batch_process_images(config_path, input_dir, output_dir):
    """
    Process all images in input directory.

    Args:
        config_path: Path to YAML configuration file
        input_dir: Directory containing input images
        output_dir: Directory to save output images

    Returns:
        Dictionary with processing metrics for each image
    """
    config = load_config(config_path)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.jfif']:
        image_files.extend(input_path.glob(ext))

    # Filter out Windows zone identifier files
    image_files = [f for f in image_files if ':Zone' not in str(f)]

    if not image_files:
        print(f"No images found in {input_dir}")
        return {}

    print(f"Found {len(image_files)} images to process\n")

    results = {}

    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_file.name}...")

        # Update config with current image
        config['input']['image'] = str(img_file)

        # Time the processing
        start_time = time.time()
        result_img = detect_lanes(config)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Extract lane information for metrics
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  ⚠ Failed to load image, skipping...")
            continue

        gray = grayscale(img)
        blurred = apply_gaussian_blur(gray, config['preprocessing']['blur_kernel'])

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

        # Save result
        output_file = output_path / f"{img_file.stem}_output.jpg"
        if result_img is not None:
            cv2.imwrite(str(output_file), result_img)
            print(f"  ✓ Saved to: {output_file.name}")
        else:
            print(f"  ⚠ Processing failed")

        # Determine detection status
        left_detected = left_lane is not None
        right_detected = right_lane is not None
        status = "✓ Both lanes" if (left_detected and right_detected) else \
                 "⚐ Left only" if left_detected else \
                 "⚑ Right only" if right_detected else \
                 "✗ No lanes"

        print(f"  {status} | {len(left_lines)} left segs, {len(right_lines)} right segs | {processing_time:.2f} ms\n")

        # Store metrics
        results[img_file.name] = {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'num_left_segments': len(left_lines),
            'num_right_segments': len(right_lines),
            'processing_time': processing_time,
            'config_used': config.copy(),
            'output_path': str(output_file),
            'input_path': str(img_file)
        }

    return results


def main():
    """Command-line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description='Batch process images for lane detection analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with default config
  python src/analysis/batch_process.py \\
      --config src/config.yaml \\
      --input data/input \\
      --output results/visualizations

  # Use highway config
  python src/analysis/batch_process.py \\
      --config src/config_highway.yaml \\
      --input data/input \\
      --output results/highway_results
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing images'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for processed images'
    )

    parser.add_argument(
        '--report',
        default=None,
        help='Generate analysis report at specified path (optional)'
    )

    parser.add_argument(
        '--csv',
        default=None,
        help='Export CSV results at specified path (optional)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Lane Detection Batch Processing")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    try:
        results = batch_process_images(args.config, args.input, args.output)

        print("=" * 60)
        print(f"Processed {len(results)} images successfully!")
        print("=" * 60)

        # Generate reports if requested
        if args.report or args.csv:
            # Add analysis to path
            analysis_path = Path(__file__).parent
            sys.path.insert(0, str(analysis_path))

            from metrics import (
                compute_detection_rate,
                analyze_failure_modes,
                compute_processing_time_stats
            )
            from reporter import generate_markdown_report, export_results_csv

            metrics = compute_detection_rate(results)
            failure_modes = analyze_failure_modes(results)
            timings = [r['processing_time'] for r in results.values()]
            timing_stats = compute_processing_time_stats(timings)

            if args.report:
                generate_markdown_report(results, metrics, failure_modes,
                                       timing_stats, args.report)

            if args.csv:
                export_results_csv(results, args.csv)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
