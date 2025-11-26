# Classical Lane Detection using Computer Vision

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-97%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-63%25-yellow)

A classical computer vision approach to lane detection using Canny edge detection and Hough Transform. This project demonstrates fundamental CV techniques before transitioning to deep learning methods, serving as the foundation for a 12-project thesis path on 3D lane detection.

## 🎯 Overview

This project implements a complete lane detection pipeline using traditional computer vision techniques:
- **Canny Edge Detection** for identifying lane boundaries
- **Hough Transform** for line detection
- **Region of Interest (ROI)** masking for focusing on relevant areas
- **Lane separation and averaging** using geometric constraints

The implementation includes a comprehensive test suite (97 tests, 63% coverage), batch processing capabilities, and performance analysis tools.

## ✨ Features

- **9-Step Detection Pipeline**: Grayscale → Blur → ROI → Edge Detection → Hough Transform → Lane Separation → Averaging → Visualization
- **97 Unit Tests**: Comprehensive test coverage with pytest
- **Batch Processing Framework**: Process multiple images with performance metrics
- **Multiple Configuration Profiles**: Default, highway, and urban scenarios
- **Analysis Tools**: Performance benchmarks, failure mode analysis, CSV export
- **CI/CD Ready**: GitHub Actions workflow for automated testing



### Detection Performance

| Metric | Value |
|--------|-------|
| **Success Rate** | 62.5% (5/8 images with both lanes detected) |
| **Average Processing Time** | 52.8 ms per image |
| **Median Processing Time** | 9.0 ms per image |
| **Test Coverage** | 63% (97 passing tests) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/classical-lane-detection.git
cd classical-lane-detection

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Run lane detection on a single image
python src/main.py --config src/config.yaml

# Use highway-optimized config
python src/main.py --config src/config_highway.yaml

# Use urban-optimized config
python src/main.py --config src/config_urban.yaml
```

### Batch Processing

Process all test images and generate analysis reports:

```bash
python src/analysis/batch_process.py \
    --config src/config.yaml \
    --input data/input \
    --output results/visualizations \
    --report docs/analysis_report.md \
    --csv results/analysis_results.csv
```

## 📊 Results Summary

**Detection Success by Category:**
- ✅ Both lanes detected: 5/8 images (62.5%)
- ⚐ Only one lane detected: 1/8 images (12.5%)
- ❌ No lanes detected: 2/8 images (25.0%)

**Performance Characteristics:**
- **Fast Processing**: 5-17ms for most images (excluding outliers)
- **Interpretable**: Clear visualization of each pipeline step
- **No Training Required**: Works out-of-the-box on new images

**Known Limitations:**
- Sensitive to lighting conditions and shadows
- Fixed parameter set may not work optimally for all scenarios
- Hard-coded slope thresholds assume specific lane orientations
- Single-frame processing (no temporal smoothing)

## 📁 Project Structure

```
classical-lane-detection/
├── src/
│   ├── preprocessing.py          # Image preprocessing functions
│   ├── edge_detection.py         # Canny edge detection
│   ├── line_detection.py         # Hough transform and lane logic
│   ├── main.py                   # Main pipeline orchestrator
│   ├── config.yaml               # Default configuration
│   ├── config_highway.yaml       # Highway-optimized config
│   ├── config_urban.yaml         # Urban-optimized config
│   └── analysis/                 # Batch processing and analysis tools
│       ├── batch_process.py      # Batch processing CLI
│       ├── metrics.py            # Performance metrics
│       ├── visualizer.py         # Visualization utilities
│       └── reporter.py           # Report generation
├── tests/
│   ├── conftest.py               # Shared test fixtures
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_edge_detection.py   # Edge detection tests
│   ├── test_line_detection.py   # Line detection tests
│   └── test_integration.py      # Integration tests
├── data/
│   ├── input/                    # Test images
│   └── output/                   # Output directory
├── results/
│   ├── visualizations/           # Processed images with lane overlays
│   └── analysis_results.csv      # Performance data
├── docs/
│   ├── README.md                 # Detailed technical documentation
│   ├── PROJECT_01_CLASSICAL_LANES.md  # Learning guide
│   └── analysis_report.md        # Performance analysis report
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=term --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

### Run Specific Test Modules

```bash
# Test preprocessing only
pytest tests/test_preprocessing.py -v

# Test line detection
pytest tests/test_line_detection.py -v

# Integration tests
pytest tests/test_integration.py -v
```

## 📚 Documentation

- **[Detailed README](docs/README.md)** - Comprehensive technical documentation
- **[Learning Guide](docs/PROJECT_01_CLASSICAL_LANES.md)** - Step-by-step project guide with 7 phases
- **[Analysis Report](docs/analysis_report.md)** - Performance benchmarks and failure analysis

## 🎓 Learning Objectives

This project helps understand:
1. Image preprocessing pipelines (grayscale, blur, ROI masking)
2. Edge detection algorithms (Canny)
3. Geometric transformations (Hough Transform)
4. Limitations of classical methods
5. Why deep learning is needed for robust lane detection

## 🔬 Algorithm Details

### 9-Step Pipeline

1. **Load Image**: Read BGR image from file
2. **Grayscale Conversion**: Convert to single-channel using `cv2.COLOR_BGR2GRAY`
3. **Gaussian Blur**: Reduce noise with configurable kernel size
4. **ROI Masking**: Apply trapezoidal mask to focus on road region
5. **Canny Edge Detection**: Detect strong edges using double thresholding
6. **Hough Transform**: Detect line segments probabilistically
7. **Lane Separation**: Classify lines as left (slope < -0.5) or right (slope > 0.5)
8. **Lane Averaging**: Fit polynomial to segments and extrapolate to image boundaries
9. **Visualization**: Overlay detected lanes on original image with blending

### Configuration Parameters

| Parameter | Default | Highway | Urban | Description |
|-----------|---------|---------|-------|-------------|
| Blur Kernel | 1 | 3 | 7 | Gaussian blur kernel size (odd) |
| Canny Low | 50 | 40 | 60 | Low threshold for edge detection |
| Canny High | 150 | 120 | 180 | High threshold for edge detection |
| Hough Threshold | 50 | 40 | 60 | Minimum votes for line detection |
| Min Line Length | 30 | 80 | 30 | Minimum line segment length (px) |
| Max Line Gap | 50 | 200 | 100 | Maximum gap to connect segments (px) |

## 🤝 Contributing

Contributions are welcome! This is a learning project from a 12-project thesis path. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Contribution Steps:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit your changes (`git commit -am 'Add improvement'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Part of a 12-project learning path for 3D Lane Detection thesis
- Built with OpenCV, NumPy, and pytest
- Test images from publicly available dashcam datasets

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Note**: This project demonstrates classical computer vision techniques as a foundation before transitioning to deep learning approaches. For production lane detection systems, consider modern deep learning methods (semantic segmentation, regression-based detection, etc.) which offer better robustness and generalization.
