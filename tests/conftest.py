"""
Shared pytest fixtures for classical lane detection tests.

This module provides reusable test data and configuration for all test modules.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import yaml


@pytest.fixture
def sample_image():
    """
    Create synthetic 640x480 BGR test image with lane markings.

    Returns:
        np.ndarray: BGR image with simulated lane lines
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw white left lane (negative slope ~-0.7)
    cv2.line(img, (200, 480), (280, 288), (255, 255, 255), 3)

    # Draw white right lane (positive slope ~0.7)
    cv2.line(img, (440, 480), (360, 288), (255, 255, 255), 3)

    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


@pytest.fixture
def sample_gray_image(sample_image):
    """
    Grayscale version of sample_image.

    Args:
        sample_image: BGR sample image fixture

    Returns:
        np.ndarray: Grayscale version of sample image
    """
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def sample_edge_image():
    """
    Binary edge image for line detection tests.

    Returns:
        np.ndarray: Binary edge image with two diagonal edges
    """
    img = np.zeros((480, 640), dtype=np.uint8)

    # Two diagonal edges representing lanes
    cv2.line(img, (200, 480), (280, 288), 255, 1)
    cv2.line(img, (440, 480), (360, 288), 255, 1)

    return img


@pytest.fixture
def sample_config(tmp_path):
    """
    Valid configuration dictionary for testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        dict: Complete configuration dictionary
    """
    test_image = tmp_path / "test.jpg"

    return {
        'input': {'image': str(test_image)},
        'output': {'path': None, 'display': False},
        'preprocessing': {'blur_kernel': 5},
        'edge_detection': {'canny_low': 50, 'canny_high': 150},
        'hough_transform': {
            'rho': 1,
            'theta': 0.017453293,  # np.pi/180
            'threshold': 50,
            'min_line_length': 30,
            'max_line_gap': 50
        },
        'roi': {'vertices': None},
        'logging': {'level': 'INFO'}
    }


@pytest.fixture
def roi_vertices():
    """
    Standard trapezoid ROI for testing.

    Returns:
        np.ndarray: ROI vertices in correct format for define_roi()
    """
    return np.array([[
        (100, 480),
        (250, 288),
        (390, 288),
        (540, 480)
    ]], dtype=np.int32)


@pytest.fixture
def sample_hough_lines():
    """
    Mock Hough line segments for lane separation tests.

    Returns:
        np.ndarray: Array of line segments in Hough format (N, 1, 4)
    """
    # Left lane lines (negative slope)
    left = np.array([
        [[200, 480, 250, 350]],
        [[210, 470, 255, 345]],
    ])

    # Right lane lines (positive slope)
    right = np.array([
        [[440, 480, 390, 350]],
        [[430, 470, 385, 345]],
    ])

    return np.vstack([left, right])


@pytest.fixture
def test_images_dir():
    """
    Path to actual test images directory.

    Returns:
        Path: Path object pointing to data/input directory
    """
    return Path(__file__).parent.parent / "data" / "input"


@pytest.fixture(params=[1, 3, 5, 7, 9])
def blur_kernel_sizes(request):
    """
    Parametrized blur kernel sizes for testing multiple values.

    Args:
        request: pytest request object

    Returns:
        int: Kernel size (always odd)
    """
    return request.param


@pytest.fixture
def temp_image_path(tmp_path, sample_image):
    """
    Temporary image file for I/O tests.

    Args:
        tmp_path: pytest temporary directory fixture
        sample_image: Sample BGR image fixture

    Returns:
        Path: Path to temporary image file
    """
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


@pytest.fixture
def blank_image():
    """
    Uniform blank image for edge case testing.

    Returns:
        np.ndarray: 100x100 uniform gray image
    """
    return np.ones((100, 100), dtype=np.uint8) * 128


@pytest.fixture
def vertical_edge_image():
    """
    Image with a single vertical edge for gradient testing.

    Returns:
        np.ndarray: 100x100 image with vertical edge at x=50
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255  # Right half is white
    return img


@pytest.fixture
def horizontal_edge_image():
    """
    Image with a single horizontal edge for gradient testing.

    Returns:
        np.ndarray: 100x100 image with horizontal edge at y=50
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[50:, :] = 255  # Bottom half is white
    return img
