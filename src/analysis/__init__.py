"""
Analysis module for lane detection performance evaluation.

This module provides tools for batch processing, metrics computation,
visualization, and report generation.
"""

from .metrics import (
    compute_detection_rate,
    analyze_failure_modes,
    compute_processing_time_stats
)

from .reporter import (
    generate_markdown_report,
    export_results_csv
)

__all__ = [
    'compute_detection_rate',
    'analyze_failure_modes',
    'compute_processing_time_stats',
    'generate_markdown_report',
    'export_results_csv'
]

__version__ = '1.0.0'
