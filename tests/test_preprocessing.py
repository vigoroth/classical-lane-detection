"""
Unit tests for preprocessing module.

Tests all preprocessing functions including grayscale conversion, blur operations,
ROI masking, and gradient computation.
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import (
    grayscale,
    apply_gaussian_blur,
    apply_bilateral_filter,
    define_roi,
    compute_gradient_image
)


class TestGrayscale:
    """Test suite for grayscale conversion."""

    def test_grayscale_basic(self, sample_image):
        """Verify BGR to grayscale conversion works."""
        gray = grayscale(sample_image)
        assert len(gray.shape) == 2, "Output should be 2D"
        assert gray.dtype == np.uint8

    def test_grayscale_preserves_shape(self, sample_image):
        """Ensure height and width are preserved."""
        gray = grayscale(sample_image)
        assert gray.shape[0] == sample_image.shape[0]
        assert gray.shape[1] == sample_image.shape[1]

    def test_grayscale_values_range(self, sample_image):
        """Check output values are in [0, 255]."""
        gray = grayscale(sample_image)
        assert np.all(gray >= 0) and np.all(gray <= 255)

    def test_grayscale_none_input(self):
        """Test handling of None input."""
        with pytest.raises((AttributeError, cv2.error)):
            grayscale(None)

    def test_grayscale_empty_image(self):
        """Test with zero-sized array."""
        empty = np.array([])
        with pytest.raises((cv2.error, ValueError, AttributeError)):
            grayscale(empty)

    def test_grayscale_consistency(self):
        """Test conversion is consistent and deterministic."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        gray1 = grayscale(img)
        gray2 = grayscale(img)
        assert np.array_equal(gray1, gray2)


class TestGaussianBlur:
    """Test suite for Gaussian blur."""

    def test_gaussian_blur_basic(self, sample_gray_image):
        """Standard blur with kernel=5."""
        blurred = apply_gaussian_blur(sample_gray_image, 5)
        assert blurred.shape == sample_gray_image.shape
        assert blurred.dtype == np.uint8

    def test_gaussian_blur_kernel_sizes(self, sample_gray_image, blur_kernel_sizes):
        """Test various odd kernel sizes."""
        blurred = apply_gaussian_blur(sample_gray_image, blur_kernel_sizes)
        assert blurred.shape == sample_gray_image.shape

    def test_gaussian_blur_smoothing_effect(self, sample_gray_image):
        """Verify blur reduces variance."""
        blurred = apply_gaussian_blur(sample_gray_image, 9)
        original_var = np.var(sample_gray_image)
        blurred_var = np.var(blurred)
        # Blurring should reduce variance (in most cases)
        # For synthetic images, check it doesn't increase dramatically
        assert blurred_var <= original_var * 1.1

    def test_gaussian_blur_invalid_kernel(self, sample_gray_image):
        """Even kernel should fail."""
        with pytest.raises(cv2.error):
            apply_gaussian_blur(sample_gray_image, 4)

    def test_gaussian_blur_zero_kernel(self, sample_gray_image):
        """kernel=0 should fail."""
        with pytest.raises(cv2.error):
            apply_gaussian_blur(sample_gray_image, 0)

    def test_gaussian_blur_negative_kernel(self, sample_gray_image):
        """Negative kernel should fail."""
        with pytest.raises(cv2.error):
            apply_gaussian_blur(sample_gray_image, -3)

    def test_gaussian_blur_color_image(self, sample_image):
        """Blur should work on color images too."""
        blurred = apply_gaussian_blur(sample_image, 5)
        assert blurred.shape == sample_image.shape
        assert len(blurred.shape) == 3


class TestBilateralFilter:
    """Test suite for bilateral filter."""

    def test_bilateral_filter_basic(self, sample_gray_image):
        """Standard bilateral filter."""
        filtered = apply_bilateral_filter(sample_gray_image)
        assert filtered.shape == sample_gray_image.shape
        assert filtered.dtype == np.uint8

    def test_bilateral_filter_custom_params(self, sample_gray_image):
        """Test with custom diameter and sigma values."""
        filtered = apply_bilateral_filter(sample_gray_image, diameter=15, sigma_color=100, sigma_space=100)
        assert filtered.shape == sample_gray_image.shape


class TestDefineROI:
    """Test suite for ROI masking."""

    def test_define_roi_basic(self, sample_gray_image, roi_vertices):
        """Standard trapezoid ROI."""
        masked = define_roi(sample_gray_image, roi_vertices)
        assert masked.shape == sample_gray_image.shape

    def test_define_roi_masks_correctly(self, sample_gray_image, roi_vertices):
        """Pixels outside ROI should be zero."""
        masked = define_roi(sample_gray_image, roi_vertices)
        # Check corner pixels (should be outside trapezoid)
        assert masked[0, 0] == 0
        assert masked[0, -1] == 0

    def test_define_roi_preserves_inside(self, roi_vertices):
        """Pixels inside ROI should be preserved."""
        # Create uniform white image
        img = np.ones((480, 640), dtype=np.uint8) * 255
        masked = define_roi(img, roi_vertices)
        # Center bottom should be inside ROI
        assert masked[450, 320] == 255

    def test_define_roi_triangle(self, sample_gray_image):
        """Test with 3-vertex polygon."""
        triangle = np.array([[(100, 480), (320, 200), (540, 480)]], dtype=np.int32)
        masked = define_roi(sample_gray_image, triangle)
        assert masked.shape == sample_gray_image.shape

    def test_define_roi_full_image(self, sample_gray_image):
        """ROI covering entire image."""
        h, w = sample_gray_image.shape
        full_roi = np.array([[(0, 0), (w, 0), (w, h), (0, h)]], dtype=np.int32)
        masked = define_roi(sample_gray_image, full_roi)
        # Should preserve entire image
        assert np.array_equal(masked, sample_gray_image)

    def test_define_roi_color_image(self, sample_image, roi_vertices):
        """ROI should work on color images too."""
        masked = define_roi(sample_image, roi_vertices)
        assert masked.shape == sample_image.shape

    def test_define_roi_small_region(self, sample_gray_image):
        """Test with very small ROI."""
        small_roi = np.array([[(300, 300), (310, 300), (310, 310), (300, 310)]], dtype=np.int32)
        masked = define_roi(sample_gray_image, small_roi)
        # Most of image should be masked (zero)
        assert np.sum(masked == 0) > (masked.size * 0.95)


class TestComputeGradient:
    """Test suite for gradient computation."""

    def test_compute_gradient_basic(self, sample_gray_image):
        """Compute magnitude and direction."""
        magnitude, direction = compute_gradient_image(sample_gray_image)
        assert magnitude.shape == sample_gray_image.shape
        assert direction.shape == sample_gray_image.shape

    def test_compute_gradient_magnitude_range(self, sample_gray_image):
        """Magnitudes should be non-negative."""
        magnitude, _ = compute_gradient_image(sample_gray_image)
        assert np.all(magnitude >= 0)

    def test_compute_gradient_direction_range(self, sample_gray_image):
        """Direction should be in [0, 360)."""
        _, direction = compute_gradient_image(sample_gray_image)
        assert np.all(direction >= 0) and np.all(direction < 360)

    def test_compute_gradient_vertical_edge(self, vertical_edge_image):
        """Vertical edge should have horizontal gradient."""
        magnitude, direction = compute_gradient_image(vertical_edge_image)
        # Around edge, direction should be ~0 or ~180 (horizontal)
        edge_directions = direction[30:70, 48:52]
        # Most should be close to 0 or 180
        horizontal = np.logical_or(
            edge_directions < 10,
            edge_directions > 170
        )
        assert np.sum(horizontal) > edge_directions.size * 0.5

    def test_compute_gradient_uniform_image(self, blank_image):
        """Uniform image should have zero gradient."""
        magnitude, _ = compute_gradient_image(blank_image)
        # Should be near zero everywhere
        assert np.max(magnitude) < 10

    def test_compute_gradient_horizontal_edge(self, horizontal_edge_image):
        """Horizontal edge should have vertical gradient."""
        magnitude, direction = compute_gradient_image(horizontal_edge_image)
        # Around edge, direction should be ~90 or ~270 (vertical)
        edge_directions = direction[48:52, 30:70]
        # Most should be close to 90 or 270
        vertical = np.logical_or(
            np.abs(edge_directions - 90) < 10,
            np.abs(edge_directions - 270) < 10
        )
        assert np.sum(vertical) > edge_directions.size * 0.5

    def test_compute_gradient_data_type(self, sample_gray_image):
        """Verify output data types."""
        magnitude, direction = compute_gradient_image(sample_gray_image)
        assert magnitude.dtype == np.float64
        assert direction.dtype == np.float64
