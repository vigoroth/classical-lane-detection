import cv2
import numpy as np
import argparse
import logging
import sys
import yaml
from pathlib import Path

from preprocessing import (
    grayscale,
    apply_gaussian_blur,
    define_roi,
)
from edge_detection import detect_edge_canny
from line_detection import (
    detect_line_hough,
    separate_lanes,
    average_lane_line,
    draw_lane_lines
)

# Initialize logger (will be configured after loading config)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.debug(f"Successfully loaded config from: {config_path}")
        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config file: {e}")


def validate_config(config):
    """
    Validate configuration dictionary.

    Args:
        config (dict): Configuration dictionary to validate

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Check required top-level sections
    required_sections = ['input', 'output', 'preprocessing', 'edge_detection',
                        'hough_transform', 'roi', 'logging']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate input section
    if 'image' not in config['input']:
        raise ValueError("Missing required field: 'input.image'")

    if config['input']['image'] is None:
        raise ValueError("Input image path cannot be null")

    # Validate preprocessing
    blur_kernel = config['preprocessing']['blur_kernel']
    if blur_kernel % 2 == 0:
        raise ValueError(f"Blur kernel must be odd, got: {blur_kernel}")

    if blur_kernel < 1:
        raise ValueError(f"Blur kernel must be positive, got: {blur_kernel}")

    # Validate edge detection thresholds
    canny_low = config['edge_detection']['canny_low']
    canny_high = config['edge_detection']['canny_high']

    if canny_low >= canny_high:
        raise ValueError(f"Canny low threshold ({canny_low}) must be less than high threshold ({canny_high})")

    if canny_low < 0 or canny_high < 0:
        raise ValueError("Canny thresholds must be non-negative")

    # Validate Hough parameters
    hough = config['hough_transform']

    if hough['rho'] <= 0:
        raise ValueError(f"Hough rho must be positive, got: {hough['rho']}")

    if hough['theta'] <= 0:
        raise ValueError(f"Hough theta must be positive, got: {hough['theta']}")

    if hough['threshold'] < 0:
        raise ValueError(f"Hough threshold must be non-negative, got: {hough['threshold']}")

    if hough['min_line_length'] < 0:
        raise ValueError(f"Hough min_line_length must be non-negative, got: {hough['min_line_length']}")

    if hough['max_line_gap'] < 0:
        raise ValueError(f"Hough max_line_gap must be non-negative, got: {hough['max_line_gap']}")

    # Validate logging level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log_level = config['logging']['level'].upper()

    if log_level not in valid_levels:
        raise ValueError(f"Invalid logging level: {log_level}. Must be one of {valid_levels}")

    logger.debug("Configuration validation passed")


def configure_logging(config):
    """
    Configure logging based on config settings.

    Args:
        config (dict): Configuration dictionary
    """
    log_level = getattr(logging, config['logging']['level'].upper())

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Reconfigure if already configured
    )

    # Update logger level
    logger.setLevel(log_level)


def detect_lanes(config):
    """
    Integrated lane detection pipeline combining all 9 steps:
    1. Load image
    2. Grayscale conversion
    3. Gaussian blur
    4. ROI mask
    5. Canny edge detection
    6. Hough line detection
    7. Lane separation
    8. Lane averaging
    9. Visualization overlay

    Args:
        config (dict): Configuration dictionary containing all parameters

    Returns:
        np.ndarray: Image with detected lanes overlaid, or None if error occurs
    """
    try:
        # Extract parameters from config
        image_path = config['input']['image']
        canny_low = config['edge_detection']['canny_low']
        canny_high = config['edge_detection']['canny_high']
        blur_kernel = config['preprocessing']['blur_kernel']
        hough_config = config['hough_transform']
        roi_vertices = config['roi']['vertices']

        # Step 1: Load image
        logger.info(f"Loading image from: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            return None

        logger.info(f"Image loaded successfully. Shape: {image.shape}")

        # Step 2: Grayscale conversion
        logger.debug("Converting image to grayscale")
        gray = grayscale(image)

        # Step 3: Gaussian blur
        logger.debug(f"Applying Gaussian blur with kernel size {blur_kernel}")
        blurred = apply_gaussian_blur(gray, blur_kernel)

        # Step 4: Define ROI mask
        if roi_vertices is None:
            # Default ROI - trapezoid covering typical lane region
            height, width = image.shape[:2]
            roi_vertices = np.array([[
                (int(width * 0.1), height),
                (int(width * 0.45), int(height * 0.6)),
                (int(width * 0.55), int(height * 0.6)),
                (int(width * 0.9), height)
            ]], dtype=np.int32)
            logger.debug(f"Using default ROI vertices")
        else:
            # Convert to numpy array if provided as list
            roi_vertices = np.array([roi_vertices], dtype=np.int32)
            logger.debug(f"Using custom ROI vertices from config")

        masked = define_roi(blurred, roi_vertices)

        # Step 5: Canny edge detection
        logger.debug(f"Applying Canny edge detection (low={canny_low}, high={canny_high})")
        edges = detect_edge_canny(masked, canny_low, canny_high)

        # Step 6: Hough line detection
        logger.debug(f"Detecting lines with Hough transform (threshold={hough_config['threshold']})")
        lines = detect_line_hough(
            edges,
            rho=hough_config['rho'],
            theta=hough_config['theta'],
            threshold=hough_config['threshold'],
            min_line_length=hough_config['min_line_length'],
            max_line_gap=hough_config['max_line_gap']
        )

        if lines is None or len(lines) == 0:
            logger.warning("No lines detected in the image")
            return image

        logger.info(f"Detected {len(lines)} line segments")

        # Step 7: Separate lanes
        logger.debug("Separating lines into left and right lanes")
        left_lines, right_lines = separate_lanes(lines, image.shape)
        logger.info(f"Left lane segments: {len(left_lines)}, Right lane segments: {len(right_lines)}")

        # Step 8: Average lane lines
        logger.debug("Averaging lane lines")
        left_lane = average_lane_line(left_lines, image.shape[0])
        right_lane = average_lane_line(right_lines, image.shape[0])

        if left_lane is None and right_lane is None:
            logger.warning("Could not compute averaged lane lines")
            return image

        # Step 9: Visualization overlay
        logger.debug("Creating visualization overlay")
        result = draw_lane_lines(image, left_lane, right_lane)

        logger.info("Lane detection completed successfully")
        return result

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except cv2.error as e:
        logger.error(f"OpenCV error during processing: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during lane detection: {e}", exc_info=True)
        return None


def main():
    """Main function to run lane detection with YAML configuration."""
    # Parse command line arguments (only for config file path)
    parser = argparse.ArgumentParser(
        description='Classical Lane Detection Pipeline (Config-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config file (src/config.yaml)
  python src/main.py

  # Use custom config file
  python src/main.py --config config_highway.yaml

  # Use config file with absolute path
  python src/main.py --config /path/to/custom_config.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='/home/vigoroth/mst_research/temporal 3D detection  BEV/learning_projects/project-01-classical-lanes/src/config.yaml',
        help='Path to YAML configuration file (default: src/config_highway.yaml)'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        # Configure logging first (before validation logging)
        configure_logging(config)

        logger.info("=" * 60)
        logger.info("Classical Lane Detection Pipeline")
        logger.info("=" * 60)

        # Validate configuration
        logger.info("Validating configuration...")
        validate_config(config)
        logger.info("Configuration validated successfully")

        # Run lane detection
        logger.info("Starting lane detection pipeline...")
        result = detect_lanes(config)

        if result is None:
            logger.error("Lane detection failed")
            return 1

        # Save output if specified
        output_path = config['output']['path']
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_file), result)
            logger.info(f"Output saved to: {output_path}")

        # Display result if enabled
        if config['output']['display']:
            cv2.imshow('Lane Detection Result', result)
            logger.info("Displaying result. Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: Configuration validation failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
