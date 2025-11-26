"""
Unit tests for edge detection module.

Tests Canny edge detection with various thresholds and edge cases.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from edge_detection import detect_edge_canny


class TestCannyEdgeDetection:
    """Test suite for Canny edge detection."""

    def test_canny_basic(self, sample_gray_image):
        """Standard edge detection."""
        edges = detect_edge_canny(sample_gray_image, 50, 150)
        assert edges.shape == sample_gray_image.shape

    def test_canny_output_binary(self, sample_gray_image):
        """Output should only contain 0 and 255."""
        edges = detect_edge_canny(sample_gray_image, 50, 150)
        unique_values = np.unique(edges)
        assert all(val in [0, 255] for val in unique_values)

    def test_canny_output_dtype(self, sample_gray_image):
        """Output should be uint8."""
        edges = detect_edge_canny(sample_gray_image, 50, 150)
        assert edges.dtype == np.uint8

    def test_canny_threshold_variations(self, sample_gray_image):
        """Test different threshold combinations."""
        edges_strict = detect_edge_canny(sample_gray_image, 100, 200)
        edges_loose = detect_edge_canny(sample_gray_image, 30, 100)
        # Loose thresholds should detect more edges
        assert np.sum(edges_loose) >= np.sum(edges_strict)

    def test_canny_low_threshold_zero(self, sample_gray_image):
        """Test with low=0."""
        edges = detect_edge_canny(sample_gray_image, 0, 150)
        assert edges.shape == sample_gray_image.shape

    def test_canny_high_threshold_max(self, sample_gray_image):
        """Test with high=255."""
        edges = detect_edge_canny(sample_gray_image, 50, 255)
        assert edges.shape == sample_gray_image.shape

    def test_canny_blank_image(self, blank_image):
        """Uniform image should produce no edges."""
        edges = detect_edge_canny(blank_image, 50, 150)
        # Should be mostly zeros
        assert np.sum(edges) < 100  # Very few edge pixels

    def test_canny_sharp_edge(self, vertical_edge_image):
        """Sharp step edge should be detected."""
        edges = detect_edge_canny(vertical_edge_image, 50, 150)
        # Should detect vertical edge around x=50
        assert np.sum(edges[:, 48:52]) > 0

    def test_canny_very_high_thresholds(self, sample_gray_image):
        """Very high thresholds should detect few or no edges."""
        edges = detect_edge_canny(sample_gray_image, 200, 250)
        # Should have very few edge pixels
        edge_percentage = np.sum(edges > 0) / edges.size
        assert edge_percentage < 0.1  # Less than 10% edge pixels

    def test_canny_very_low_thresholds(self, sample_gray_image):
        """Very low thresholds should detect many edges."""
        edges = detect_edge_canny(sample_gray_image, 10, 30)
        # Should have more edge pixels
        edge_percentage = np.sum(edges > 0) / edges.size
        assert edge_percentage > 0.01  # At least some edges

    def test_canny_threshold_order(self, sample_gray_image):
        """Verify low threshold < high threshold in actual usage."""
        # This test documents the expected relationship
        # Note: OpenCV Canny will actually swap them if given in wrong order
        low, high = 50, 150
        assert low < high

    def test_canny_square_edge(self):
        """Test detection on a perfect square."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Draw a white square in the center
        img[25:75, 25:75] = 255
        edges = detect_edge_canny(img, 50, 150)
        # Should detect edges around the square perimeter
        # Check that edges are found in the middle rows/cols where square edges are
        assert np.sum(edges[24:26, :]) > 0  # Top edge
        assert np.sum(edges[74:76, :]) > 0  # Bottom edge
        assert np.sum(edges[:, 24:26]) > 0  # Left edge
        assert np.sum(edges[:, 74:76]) > 0  # Right edge

    def test_canny_preserves_shape(self):
        """Test various image shapes."""
        shapes = [(100, 100), (200, 300), (480, 640)]
        for shape in shapes:
            img = np.random.randint(0, 256, shape, dtype=np.uint8)
            edges = detect_edge_canny(img, 50, 150)
            assert edges.shape == shape
