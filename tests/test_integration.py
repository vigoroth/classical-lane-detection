"""
Integration tests for the full lane detection pipeline.

Tests configuration management and end-to-end pipeline execution.
"""

import pytest
import numpy as np
import cv2
import yaml
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from main import load_config, validate_config, detect_lanes, configure_logging


class TestConfigurationManagement:
    """Test config loading and validation."""

    def test_load_config_valid(self, tmp_path):
        """Load valid YAML config."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            'input': {'image': 'test.jpg'},
            'output': {'path': None, 'display': False},
            'preprocessing': {'blur_kernel': 5},
            'edge_detection': {'canny_low': 50, 'canny_high': 150},
            'hough_transform': {
                'rho': 1, 'theta': 0.017453293, 'threshold': 50,
                'min_line_length': 30, 'max_line_gap': 50
            },
            'roi': {'vertices': None},
            'logging': {'level': 'INFO'}
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config['preprocessing']['blur_kernel'] == 5

    def test_load_config_missing_file(self):
        """FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')

    def test_validate_config_even_blur_kernel(self, sample_config):
        """Even blur kernel should fail."""
        sample_config['preprocessing']['blur_kernel'] = 4
        with pytest.raises(ValueError, match="Blur kernel must be odd"):
            validate_config(sample_config)

    def test_validate_config_negative_blur_kernel(self, sample_config):
        """Negative blur kernel should fail."""
        sample_config['preprocessing']['blur_kernel'] = -3
        with pytest.raises(ValueError, match="Blur kernel must be positive"):
            validate_config(sample_config)

    def test_validate_config_inverted_canny(self, sample_config):
        """Canny low >= high should fail."""
        sample_config['edge_detection']['canny_low'] = 200
        sample_config['edge_detection']['canny_high'] = 100
        with pytest.raises(ValueError, match="must be less than high"):
            validate_config(sample_config)

    def test_validate_config_negative_canny(self, sample_config):
        """Negative Canny threshold should fail."""
        sample_config['edge_detection']['canny_low'] = -10
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_config(sample_config)

    def test_validate_config_missing_section(self, sample_config):
        """Missing required section should fail."""
        del sample_config['preprocessing']
        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config(sample_config)

    def test_validate_config_invalid_log_level(self, sample_config):
        """Invalid logging level should fail."""
        sample_config['logging']['level'] = 'INVALID'
        with pytest.raises(ValueError, match="Invalid logging level"):
            validate_config(sample_config)

    def test_validate_config_null_image_path(self, sample_config):
        """Null image path should fail."""
        sample_config['input']['image'] = None
        with pytest.raises(ValueError, match="cannot be null"):
            validate_config(sample_config)

    def test_validate_config_negative_hough_params(self, sample_config):
        """Negative Hough parameters should fail."""
        sample_config['hough_transform']['rho'] = -1
        with pytest.raises(ValueError, match="rho must be positive"):
            validate_config(sample_config)

        sample_config['hough_transform']['rho'] = 1
        sample_config['hough_transform']['theta'] = -0.1
        with pytest.raises(ValueError, match="theta must be positive"):
            validate_config(sample_config)

    def test_validate_config_success(self, sample_config, temp_image_path):
        """Valid config should pass validation."""
        sample_config['input']['image'] = str(temp_image_path)
        # Should not raise any exception
        validate_config(sample_config)

    def test_configure_logging(self, sample_config):
        """Test logging configuration."""
        configure_logging(sample_config)
        # Should execute without errors
        # Actual verification would require checking logger state


class TestDetectLanesPipeline:
    """Test full pipeline execution."""

    def test_detect_lanes_full_pipeline(self, sample_config, temp_image_path):
        """Run complete 9-step process."""
        sample_config['input']['image'] = str(temp_image_path)
        result = detect_lanes(sample_config)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # Color image

    def test_detect_lanes_missing_image(self, sample_config):
        """Handle missing image file."""
        sample_config['input']['image'] = '/nonexistent/image.jpg'
        result = detect_lanes(sample_config)
        assert result is None

    def test_detect_lanes_with_custom_roi(self, sample_config, temp_image_path):
        """Test with custom ROI vertices."""
        sample_config['input']['image'] = str(temp_image_path)
        sample_config['roi']['vertices'] = [
            [100, 480],
            [250, 288],
            [390, 288],
            [540, 480]
        ]
        result = detect_lanes(sample_config)
        assert result is not None

    def test_detect_lanes_default_roi(self, sample_config, temp_image_path):
        """Test with default ROI (None)."""
        sample_config['input']['image'] = str(temp_image_path)
        sample_config['roi']['vertices'] = None
        result = detect_lanes(sample_config)
        assert result is not None

    def test_detect_lanes_returns_original_on_no_lines(self, sample_config, tmp_path):
        """If no lines detected, should return original image."""
        # Create blank image (no lanes)
        blank_img_path = tmp_path / "blank.jpg"
        blank_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(blank_img_path), blank_img)

        sample_config['input']['image'] = str(blank_img_path)
        result = detect_lanes(sample_config)

        # Should return the original image
        assert result is not None
        assert result.shape == blank_img.shape

    def test_detect_lanes_real_images(self, test_images_dir, sample_config):
        """Test on actual test images."""
        jpg_images = list(test_images_dir.glob('*.jpg'))
        jpg_images = [img for img in jpg_images if ':Zone' not in str(img)]

        assert len(jpg_images) > 0, "No test images found"

        for img_path in jpg_images[:3]:  # Test first 3
            sample_config['input']['image'] = str(img_path)
            result = detect_lanes(sample_config)
            # Should return something (even if no lanes detected)
            assert result is not None
            # Should be color image
            assert len(result.shape) == 3

    def test_detect_lanes_different_blur_kernels(self, sample_config, temp_image_path):
        """Test pipeline with different blur kernel sizes."""
        sample_config['input']['image'] = str(temp_image_path)

        for kernel in [1, 3, 5, 7, 9]:
            sample_config['preprocessing']['blur_kernel'] = kernel
            result = detect_lanes(sample_config)
            assert result is not None

    def test_detect_lanes_different_canny_thresholds(self, sample_config, temp_image_path):
        """Test pipeline with different Canny thresholds."""
        sample_config['input']['image'] = str(temp_image_path)

        threshold_pairs = [(30, 90), (50, 150), (70, 210)]
        for low, high in threshold_pairs:
            sample_config['edge_detection']['canny_low'] = low
            sample_config['edge_detection']['canny_high'] = high
            result = detect_lanes(sample_config)
            assert result is not None

    def test_detect_lanes_pipeline_steps_execute(self, sample_config, temp_image_path, monkeypatch):
        """Verify all pipeline steps are executed."""
        sample_config['input']['image'] = str(temp_image_path)

        # Track which functions are called
        calls = []

        # Patch preprocessing functions
        from preprocessing import grayscale as orig_grayscale
        from preprocessing import apply_gaussian_blur as orig_blur
        from preprocessing import define_roi as orig_roi
        from edge_detection import detect_edge_canny as orig_canny
        from line_detection import detect_line_hough as orig_hough

        def tracked_grayscale(img):
            calls.append('grayscale')
            return orig_grayscale(img)

        def tracked_blur(img, k):
            calls.append('blur')
            return orig_blur(img, k)

        def tracked_roi(img, v):
            calls.append('roi')
            return orig_roi(img, v)

        def tracked_canny(img, l, h):
            calls.append('canny')
            return orig_canny(img, l, h)

        def tracked_hough(img, **kwargs):
            calls.append('hough')
            return orig_hough(img, **kwargs)

        monkeypatch.setattr('main.grayscale', tracked_grayscale)
        monkeypatch.setattr('main.apply_gaussian_blur', tracked_blur)
        monkeypatch.setattr('main.define_roi', tracked_roi)
        monkeypatch.setattr('main.detect_edge_canny', tracked_canny)
        monkeypatch.setattr('main.detect_line_hough', tracked_hough)

        result = detect_lanes(sample_config)

        # Verify all steps were called
        assert 'grayscale' in calls
        assert 'blur' in calls
        assert 'roi' in calls
        assert 'canny' in calls
        assert 'hough' in calls
