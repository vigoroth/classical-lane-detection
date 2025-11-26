"""
Performance metrics for lane detection analysis.

Functions for computing detection rates, failure modes, and timing statistics.
"""

import numpy as np
from typing import Dict, List


def compute_detection_rate(results: Dict) -> Dict:
    """
    Calculate lane detection success rate.

    Args:
        results: Dictionary from batch_process_images() with format:
                {image_name: {
                    'left_lane': np.ndarray or None,
                    'right_lane': np.ndarray or None,
                    'num_left_segments': int,
                    'num_right_segments': int,
                    'processing_time': float,
                    ...
                }}

    Returns:
        Dictionary with detection statistics:
        {
            'total_images': int,
            'left_lane_detected': int,
            'right_lane_detected': int,
            'both_lanes_detected': int,
            'detection_rate': float (percentage),
            'at_least_one_detected': int
        }
    """
    total = len(results)

    if total == 0:
        return {
            'total_images': 0,
            'left_lane_detected': 0,
            'right_lane_detected': 0,
            'both_lanes_detected': 0,
            'detection_rate': 0.0,
            'at_least_one_detected': 0
        }

    left_detected = sum(1 for r in results.values() if r['left_lane'] is not None)
    right_detected = sum(1 for r in results.values() if r['right_lane'] is not None)
    both_detected = sum(
        1 for r in results.values()
        if r['left_lane'] is not None and r['right_lane'] is not None
    )
    at_least_one = sum(
        1 for r in results.values()
        if r['left_lane'] is not None or r['right_lane'] is not None
    )

    return {
        'total_images': total,
        'left_lane_detected': left_detected,
        'right_lane_detected': right_detected,
        'both_lanes_detected': both_detected,
        'detection_rate': (both_detected / total * 100) if total > 0 else 0.0,
        'at_least_one_detected': at_least_one
    }


def analyze_failure_modes(results: Dict) -> Dict[str, List[str]]:
    """
    Categorize failure cases.

    Args:
        results: Dictionary from batch_process_images()

    Returns:
        Dictionary mapping failure mode to list of image names:
        {
            'no_lines_detected': [...],
            'only_left_detected': [...],
            'only_right_detected': [...],
            'success': [...]
        }
    """
    failure_modes = {
        'no_lines_detected': [],
        'only_left_detected': [],
        'only_right_detected': [],
        'success': []
    }

    for img_name, data in results.items():
        left = data['left_lane'] is not None
        right = data['right_lane'] is not None

        if not left and not right:
            failure_modes['no_lines_detected'].append(img_name)
        elif left and not right:
            failure_modes['only_left_detected'].append(img_name)
        elif right and not left:
            failure_modes['only_right_detected'].append(img_name)
        else:
            failure_modes['success'].append(img_name)

    return failure_modes


def compute_processing_time_stats(timings: List[float]) -> Dict:
    """
    Calculate timing statistics.

    Args:
        timings: List of processing times in milliseconds

    Returns:
        Dictionary with timing statistics:
        {
            'mean': float,
            'median': float,
            'std': float,
            'min': float,
            'max': float
        }
    """
    if not timings:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

    return {
        'mean': float(np.mean(timings)),
        'median': float(np.median(timings)),
        'std': float(np.std(timings)),
        'min': float(np.min(timings)),
        'max': float(np.max(timings))
    }


def compute_segment_statistics(results: Dict) -> Dict:
    """
    Calculate statistics about detected line segments.

    Args:
        results: Dictionary from batch_process_images()

    Returns:
        Dictionary with segment statistics
    """
    left_segments = [r['num_left_segments'] for r in results.values()]
    right_segments = [r['num_right_segments'] for r in results.values()]

    return {
        'left_segments': {
            'mean': float(np.mean(left_segments)) if left_segments else 0.0,
            'median': float(np.median(left_segments)) if left_segments else 0.0,
            'max': int(np.max(left_segments)) if left_segments else 0
        },
        'right_segments': {
            'mean': float(np.mean(right_segments)) if right_segments else 0.0,
            'median': float(np.median(right_segments)) if right_segments else 0.0,
            'max': int(np.max(right_segments)) if right_segments else 0
        }
    }
