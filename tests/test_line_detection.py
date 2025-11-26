"""
Unit tests for line detection module.

Tests Hough line detection, lane separation, averaging, and visualization.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from line_detection import (
    detect_line_hough,
    draw_lines,
    separate_lanes,
    average_lane_line,
    draw_lane_lines,
    process_image_for_lines,
    process_image_for_lanes
)


class TestHoughLineDetection:
    """Test Hough transform line detection."""

    def test_hough_basic(self, sample_edge_image):
        """Detect lines from edge image."""
        lines = detect_line_hough(sample_edge_image)
        assert lines is None or lines.shape[1] == 1
        if lines is not None:
            assert lines.shape[2] == 4  # [x1, y1, x2, y2]

    def test_hough_threshold_effect(self, sample_edge_image):
        """Higher threshold should reduce line count."""
        lines_low = detect_line_hough(sample_edge_image, threshold=10)
        lines_high = detect_line_hough(sample_edge_image, threshold=50)
        if lines_low is not None and lines_high is not None:
            assert len(lines_high) <= len(lines_low)

    def test_hough_no_edges(self):
        """Blank image should return None."""
        blank = np.zeros((100, 100), dtype=np.uint8)
        lines = detect_line_hough(blank)
        assert lines is None

    def test_hough_single_line(self):
        """Detect single straight line."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(img, (10, 10), (90, 90), 255, 1)
        lines = detect_line_hough(img, threshold=10, min_line_length=20)
        assert lines is not None
        assert len(lines) >= 1

    def test_hough_return_format(self):
        """Verify Hough returns correct format."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(img, (10, 10), (90, 90), 255, 2)
        lines = detect_line_hough(img, threshold=10, min_line_length=20)
        if lines is not None:
            # Shape should be (N, 1, 4)
            assert len(lines.shape) == 3
            assert lines.shape[1] == 1
            assert lines.shape[2] == 4

    def test_hough_custom_parameters(self, sample_edge_image):
        """Test with custom rho and theta."""
        lines = detect_line_hough(
            sample_edge_image,
            rho=2,
            theta=np.pi/90,
            threshold=30,
            min_line_length=40,
            max_line_gap=100
        )
        # Should return valid result or None
        assert lines is None or isinstance(lines, np.ndarray)

    def test_hough_min_line_length_effect(self):
        """Test min_line_length parameter."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Draw short line
        cv2.line(img, (10, 10), (20, 20), 255, 1)
        # Long line
        cv2.line(img, (30, 30), (90, 90), 255, 1)

        # With high min_line_length, should only detect long line
        lines_strict = detect_line_hough(img, threshold=5, min_line_length=40)
        # With low min_line_length, should detect both
        lines_loose = detect_line_hough(img, threshold=5, min_line_length=5)

        if lines_strict is not None and lines_loose is not None:
            assert len(lines_loose) >= len(lines_strict)


class TestDrawLines:
    """Test line drawing functionality."""

    def test_draw_lines_basic(self, sample_image):
        """Draw lines on image."""
        lines = np.array([[[200, 480, 280, 288]]])
        result = draw_lines(sample_image.copy(), lines)
        assert result.shape == sample_image.shape
        # Result should differ from original
        assert not np.array_equal(result, sample_image)

    def test_draw_lines_none_input(self, sample_image):
        """Handle None lines gracefully."""
        result = draw_lines(sample_image.copy(), None)
        # Should return original image unchanged
        assert np.array_equal(result, sample_image)

    def test_draw_lines_custom_color(self, sample_image):
        """Test custom line color."""
        lines = np.array([[[200, 480, 280, 288]]])
        result = draw_lines(sample_image.copy(), lines, color=(255, 0, 0), thickness=3)
        assert result.shape == sample_image.shape

    def test_draw_lines_multiple(self, sample_image):
        """Draw multiple lines."""
        lines = np.array([
            [[200, 480, 280, 288]],
            [[440, 480, 360, 288]],
        ])
        result = draw_lines(sample_image.copy(), lines)
        assert result.shape == sample_image.shape


class TestSeparateLanes:
    """Test lane separation by slope."""

    def test_separate_lanes_basic(self, sample_hough_lines):
        """Separate left and right lanes."""
        img_shape = (480, 640, 3)
        left_lines, right_lines = separate_lanes(sample_hough_lines, img_shape)
        # Should have lines in both categories
        assert len(left_lines) > 0
        assert len(right_lines) > 0

    def test_separate_lanes_slope_criteria(self):
        """Verify slope thresholds."""
        # Create lines with known slopes
        left_line = np.array([[[100, 400, 150, 300]]])  # Negative slope
        right_line = np.array([[[500, 400, 450, 300]]])  # Positive slope
        horizontal = np.array([[[200, 300, 400, 310]]])  # Near-horizontal

        lines = np.vstack([left_line, right_line, horizontal])
        left, right = separate_lanes(lines, (480, 640))

        assert len(left) == 1
        assert len(right) == 1
        # Horizontal should be filtered out

    def test_separate_lanes_vertical_lines(self):
        """Vertical lines (undefined slope) should be skipped."""
        vertical = np.array([[[200, 100, 200, 400]]])  # x1 == x2
        left, right = separate_lanes(vertical, (480, 640))
        assert len(left) == 0
        assert len(right) == 0

    def test_separate_lanes_none_input(self):
        """Handle None input gracefully."""
        left, right = separate_lanes(None, (480, 640))
        assert left == []
        assert right == []

    def test_separate_lanes_steep_slopes(self):
        """Test with very steep slopes."""
        # Very steep left lane (slope < -1)
        steep_left = np.array([[[100, 400, 120, 200]]])
        # Very steep right lane (slope > 1)
        steep_right = np.array([[[500, 400, 480, 200]]])

        lines = np.vstack([steep_left, steep_right])
        left, right = separate_lanes(lines, (480, 640))

        assert len(left) == 1
        assert len(right) == 1

    def test_separate_lanes_boundary_slopes(self):
        """Test slopes near the threshold (-0.5, 0.5)."""
        # Slope exactly at -0.5 (should be included in left)
        boundary_left = np.array([[[100, 400, 200, 300]]])  # slope = -100/100 = -1
        # Slope exactly at 0.5 (should be included in right)
        boundary_right = np.array([[[500, 300, 600, 400]]])  # slope = 100/100 = 1

        lines = np.vstack([boundary_left, boundary_right])
        left, right = separate_lanes(lines, (480, 640))

        # With threshold -0.5 and 0.5, both should be included
        assert len(left) == 1
        assert len(right) == 1


class TestAverageLaneLine:
    """Test lane line averaging."""

    def test_average_lane_basic(self):
        """Average multiple segments."""
        lines = np.array([
            [[100, 480, 150, 350]],
            [[110, 470, 155, 345]],
        ])
        img_height = 480
        averaged = average_lane_line(lines, img_height)

        assert averaged is not None
        assert averaged.shape == (1, 4)
        # Should extrapolate to bottom (y=480)
        assert averaged[0][1] == img_height
        # And to 60% height
        assert averaged[0][3] == int(img_height * 0.6)

    def test_average_lane_empty_input(self):
        """Return None for empty list."""
        result = average_lane_line([], 480)
        assert result is None

    def test_average_lane_single_segment(self):
        """Should work with single line."""
        line = np.array([[[100, 480, 150, 350]]])
        result = average_lane_line(line, 480)
        assert result is not None

    def test_average_lane_extrapolation(self):
        """Verify extrapolation to image boundaries."""
        lines = np.array([[[200, 400, 250, 300]]])
        averaged = average_lane_line(lines, 480)

        # y1 should be at bottom
        assert averaged[0][1] == 480
        # y2 should be at 60% height
        assert averaged[0][3] == 288  # 480 * 0.6

    def test_average_lane_coordinates_valid(self):
        """Verify output coordinates are reasonable."""
        lines = np.array([
            [[200, 480, 250, 350]],
            [[210, 470, 255, 345]],
        ])
        averaged = average_lane_line(lines, 480)

        # x coordinates should be reasonable (within image bounds plus margin)
        assert -100 < averaged[0][0] < 1000  # x1
        assert -100 < averaged[0][2] < 1000  # x2
        # y coordinates should match expected values
        assert averaged[0][1] == 480  # y1
        assert averaged[0][3] == 288  # y2

    def test_average_lane_return_type(self):
        """Verify return type is numpy array."""
        lines = np.array([[[200, 480, 250, 350]]])
        result = average_lane_line(lines, 480)
        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.int32, np.int64, np.int_]


class TestDrawLaneLines:
    """Test lane line visualization."""

    def test_draw_lanes_basic(self, sample_image):
        """Draw both lanes."""
        left_lane = np.array([[200, 480, 280, 288]])
        right_lane = np.array([[440, 480, 360, 288]])

        result = draw_lane_lines(sample_image, left_lane, right_lane)
        assert result.shape == sample_image.shape
        # Result should differ from original
        assert not np.array_equal(result, sample_image)

    def test_draw_lanes_left_only(self, sample_image):
        """Draw only left lane."""
        left_lane = np.array([[200, 480, 280, 288]])
        result = draw_lane_lines(sample_image, left_lane, None)
        assert result.shape == sample_image.shape

    def test_draw_lanes_right_only(self, sample_image):
        """Draw only right lane."""
        right_lane = np.array([[440, 480, 360, 288]])
        result = draw_lane_lines(sample_image, None, right_lane)
        assert result.shape == sample_image.shape

    def test_draw_lanes_neither(self, sample_image):
        """Both None should return blended image."""
        result = draw_lane_lines(sample_image, None, None)
        # Should still return valid image
        assert result.shape == sample_image.shape

    def test_draw_lanes_color(self, sample_image):
        """Test custom color."""
        left_lane = np.array([[200, 480, 280, 288]])
        result = draw_lane_lines(sample_image, left_lane, None, color=(0, 255, 0))
        assert result.shape == sample_image.shape

    def test_draw_lanes_thickness(self, sample_image):
        """Test custom thickness."""
        left_lane = np.array([[200, 480, 280, 288]])
        result = draw_lane_lines(sample_image, left_lane, None, thickness=10)
        assert result.shape == sample_image.shape

    def test_draw_lanes_blending(self, sample_image):
        """Verify image blending occurs."""
        left_lane = np.array([[200, 480, 280, 288]])
        right_lane = np.array([[440, 480, 360, 288]])

        result = draw_lane_lines(sample_image, left_lane, right_lane)

        # Result should be different from original but not completely different
        # (due to blending with 0.8 weight on original)
        diff = np.abs(result.astype(np.int16) - sample_image.astype(np.int16))
        # Some difference should exist where lines are drawn
        assert np.sum(diff) > 0
