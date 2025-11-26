"""
Report generation utilities.

Functions for creating markdown reports and CSV exports.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict


def generate_markdown_report(results: Dict, metrics: Dict, failure_modes: Dict,
                            timing_stats: Dict, output_path: str):
    """
    Generate comprehensive Markdown analysis report.

    Args:
        results: Dictionary from batch_process_images()
        metrics: Dictionary from compute_detection_rate()
        failure_modes: Dictionary from analyze_failure_modes()
        timing_stats: Dictionary from compute_processing_time_stats()
        output_path: Where to save the report

    Returns:
        None (saves to file)
    """
    # Get one result to extract config
    sample_result = list(results.values())[0] if results else None

    report = f"""# Lane Detection Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Images Tested:** {metrics['total_images']}
- **Detection Success Rate:** {metrics['detection_rate']:.1f}%
- **Both Lanes Detected:** {metrics['both_lanes_detected']} images
- **At Least One Lane:** {metrics['at_least_one_detected']} images
- **Average Processing Time:** {timing_stats['mean']:.2f} ms
- **Median Processing Time:** {timing_stats['median']:.2f} ms

## Methodology

### Algorithm Overview

Classical computer vision approach using a 9-step pipeline:

1. **Image Loading**: Read BGR image from file
2. **Grayscale Conversion**: Convert to single-channel using OpenCV's COLOR_BGR2GRAY
3. **Gaussian Blur**: Reduce noise with configurable kernel
4. **Region of Interest**: Apply trapezoidal mask to focus on road region
5. **Canny Edge Detection**: Detect strong edges using double thresholding
6. **Hough Transform**: Detect line segments probabilistically
7. **Lane Separation**: Classify lines as left (slope < -0.5) or right (slope > 0.5)
8. **Lane Averaging**: Fit polynomial to segments and extrapolate
9. **Visualization**: Overlay lanes on original image with blending

### Parameters Used

"""

    if sample_result and 'config_used' in sample_result:
        config = sample_result['config_used']
        report += f"""| Parameter | Value |
|-----------|-------|
| Blur Kernel | {config['preprocessing']['blur_kernel']} |
| Canny Low Threshold | {config['edge_detection']['canny_low']} |
| Canny High Threshold | {config['edge_detection']['canny_high']} |
| Hough Rho | {config['hough_transform']['rho']} |
| Hough Theta | {config['hough_transform']['theta']:.6f} rad |
| Hough Threshold | {config['hough_transform']['threshold']} |
| Min Line Length | {config['hough_transform']['min_line_length']} px |
| Max Line Gap | {config['hough_transform']['max_line_gap']} px |

"""

    report += f"""## Performance Metrics

### Detection Rates

- **Left Lane Detected:** {metrics['left_lane_detected']}/{metrics['total_images']} ({metrics['left_lane_detected']/metrics['total_images']*100:.1f}%)
- **Right Lane Detected:** {metrics['right_lane_detected']}/{metrics['total_images']} ({metrics['right_lane_detected']/metrics['total_images']*100:.1f}%)
- **Both Lanes:** {metrics['both_lanes_detected']}/{metrics['total_images']} ({metrics['detection_rate']:.1f}%)

### Failure Mode Distribution

- **Success (Both Lanes):** {len(failure_modes['success'])} images
- **Only Left Detected:** {len(failure_modes['only_left_detected'])} images
- **Only Right Detected:** {len(failure_modes['only_right_detected'])} images
- **No Lines Detected:** {len(failure_modes['no_lines_detected'])} images

### Processing Time Statistics

- **Mean:** {timing_stats['mean']:.2f} ms
- **Median:** {timing_stats['median']:.2f} ms
- **Std Dev:** {timing_stats['std']:.2f} ms
- **Min:** {timing_stats['min']:.2f} ms
- **Max:** {timing_stats['max']:.2f} ms

## Results by Image

| Image | Left Lane | Right Lane | Left Segments | Right Segments | Time (ms) |
|-------|-----------|------------|---------------|----------------|-----------|
"""

    for img_name, data in sorted(results.items()):
        left_status = "✅" if data['left_lane'] is not None else "❌"
        right_status = "✅" if data['right_lane'] is not None else "❌"

        report += f"| {img_name} | {left_status} | {right_status} | {data['num_left_segments']} | {data['num_right_segments']} | {data['processing_time']:.2f} |\n"

    report += f"""

## Detailed Failure Analysis

### Successful Detections ({len(failure_modes['success'])} images)

"""
    if failure_modes['success']:
        for img in failure_modes['success']:
            report += f"- {img}\n"
    else:
        report += "*No successful detections*\n"

    report += f"""
### Only Left Lane Detected ({len(failure_modes['only_left_detected'])} images)

"""
    if failure_modes['only_left_detected']:
        for img in failure_modes['only_left_detected']:
            report += f"- {img}\n"
    else:
        report += "*None*\n"

    report += f"""
### Only Right Lane Detected ({len(failure_modes['only_right_detected'])} images)

"""
    if failure_modes['only_right_detected']:
        for img in failure_modes['only_right_detected']:
            report += f"- {img}\n"
    else:
        report += "*None*\n"

    report += f"""
### No Lines Detected ({len(failure_modes['no_lines_detected'])} images)

"""
    if failure_modes['no_lines_detected']:
        for img in failure_modes['no_lines_detected']:
            report += f"- {img}\n"
    else:
        report += "*None*\n"

    report += """
## Key Findings

### Strengths

- Fast processing time (typically < 100ms per image)
- Simple, interpretable algorithm
- No training data required
- Works well on clear, well-marked lanes

### Limitations Identified

- **Sensitivity to lighting:** Shadows and varying illumination affect edge detection
- **Fixed parameters:** Single parameter set may not work for all conditions
- **Hard-coded slope thresholds:** Assumes specific lane orientations
- **No temporal smoothing:** Single-frame processing leads to instability
- **ROI constraints:** Fixed trapezoidal ROI may miss curved lanes

## Recommendations

### Short-term Improvements

1. **Adaptive Thresholding:** Implement automatic Canny threshold selection based on image statistics
2. **Dynamic ROI:** Adjust ROI based on detected lane positions
3. **Bilateral Filtering:** Use edge-preserving smoothing for challenging lighting
4. **Parameter Profiles:** Create multiple config profiles for different scenarios (highway, urban, night)

### Medium-term Enhancements

1. **Temporal Smoothing:** For video processing, add inter-frame averaging
2. **Curved Lane Support:** Use higher-order polynomial fitting (quadratic)
3. **Confidence Scoring:** Compute and report detection confidence
4. **Outlier Rejection:** Use RANSAC for robust line fitting

### Long-term Direction

1. **Hybrid Approach:** Combine classical CV with learned features
2. **Deep Learning:** Transition to semantic segmentation or regression-based methods
3. **Multi-task Learning:** Simultaneous lane and object detection
4. **3D Lane Modeling:** Incorporate IPM (Inverse Perspective Mapping)

## Conclusion

The classical computer vision approach provides a solid baseline for lane detection with fast processing times and interpretable results. However, it shows clear limitations in handling varying lighting conditions and complex scenarios. The analysis reveals a detection success rate of **{metrics['detection_rate']:.1f}%**, indicating room for improvement.

Key takeaways:
- Classical methods are suitable for controlled environments
- Parameter tuning is critical for performance
- Modern learning-based methods are needed for robust real-world deployment

## Next Steps

1. Implement recommended short-term improvements
2. Test on larger, more diverse dataset
3. Benchmark against learning-based methods
4. Proceed to Project 2: Perspective Transformation

---

*Report generated by Classical Lane Detection Analysis Framework v1.0*
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Markdown report saved to: {output_path}")


def export_results_csv(results: Dict, output_path: str):
    """
    Export results to CSV for spreadsheet analysis.

    Args:
        results: Dictionary from batch_process_images()
        output_path: Where to save the CSV

    Returns:
        None (saves to file)
    """
    rows = []
    for img_name, data in results.items():
        row = {
            'image_name': img_name,
            'left_detected': data['left_lane'] is not None,
            'right_detected': data['right_lane'] is not None,
            'both_detected': data['left_lane'] is not None and data['right_lane'] is not None,
            'num_left_segments': data['num_left_segments'],
            'num_right_segments': data['num_right_segments'],
            'processing_time_ms': data['processing_time']
        }

        # Add config parameters
        if 'config_used' in data:
            config = data['config_used']
            row.update({
                'blur_kernel': config['preprocessing']['blur_kernel'],
                'canny_low': config['edge_detection']['canny_low'],
                'canny_high': config['edge_detection']['canny_high'],
                'hough_threshold': config['hough_transform']['threshold'],
                'min_line_length': config['hough_transform']['min_line_length'],
                'max_line_gap': config['hough_transform']['max_line_gap']
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV export saved to: {output_path}")
