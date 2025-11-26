# Project 1: Classical Lane Detection
## Understanding Computer Vision Fundamentals

**Duration:** Week 1 (8-12 hours)
**Difficulty:** ⭐⭐☆☆☆
**Prerequisites:** Basic Python, basic linear algebra
**Goal:** Build lane detector using classical CV to understand why deep learning is needed

---

## 🎯 Learning Objectives

By completing this project, you will:
1. Understand image preprocessing pipelines
2. Master edge detection algorithms (Canny)
3. Learn geometric transformations (Hough Transform)
4. Recognize limitations of classical methods
5. Build intuition for what makes lane detection hard

---

## 📋 Project Structure

Create this directory structure:
```
mini-project-01-classical-lanes/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── line_detection.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── data/
│   ├── input/
│   │   └── (place test images here)
│   └── output/
├── results/
│   └── visualizations/
└── docs/
    └── methodology.md
```

---

## 🚀 Step-by-Step Instructions

### **PHASE 1: Setup & Data (1-2 hours)**

#### Step 1.1: Environment Setup
**Task:** Create isolated Python environment

**Instructions:**
1. Create new conda environment or venv
2. Install required packages:
   - opencv-python (for image processing)
   - numpy (for numerical operations)
   - matplotlib (for visualization)
   - pytest (for testing)

**Questions to Answer:**
- Why use a virtual environment?
- What version of OpenCV should you use?
- How do you verify installation?

**Deliverable:** Working environment with all imports successful

---

#### Step 1.2: Collect Test Data
**Task:** Gather diverse lane images

**Instructions:**
1. Find 5-7 test images with lanes:
   - Highway (straight lanes, clear markings)
   - Urban (curved lanes, shadows)
   - Night (low contrast)
   - Challenging (faded markings, occlusions)

**Sources:**
- Google Images (search "highway lane")
- Dashcam videos (extract frames)
- KITTI dataset (download sample)
- Your own dashcam/phone

**Deliverable:** data/input/ folder with labeled test images

---

#### Step 1.3: Manual Analysis
**Task:** Understand what makes lanes detectable

**Instructions:**
1. For each image, manually identify:
   - Where are the lanes?
   - What makes them visible? (color, texture, edges)
   - What challenges exist? (shadows, glare, other markings)
2. Sketch expected output
3. Note failure predictions

**Deliverable:** docs/methodology.md with analysis

---

### **PHASE 2: Preprocessing Pipeline (2-3 hours)**

#### Step 2.1: Color Space Conversion
**Task:** Implement grayscale conversion

**Instructions:**
1. Create function: `convert_to_grayscale(image)`
2. Understand RGB to grayscale formula:
   - Luminosity method: 0.299R + 0.587G + 0.114B
   - Why these weights? (human perception)
3. Implement using OpenCV's `cvtColor()`
4. Visualize original vs grayscale

**Key Concept:** Why grayscale?
- Reduces computation (1 channel vs 3)
- Lanes are usually distinguishable by intensity
- Edge detection works on intensity gradients

**Test:** Apply to all test images, verify output

---

#### Step 2.2: Noise Reduction
**Task:** Implement Gaussian blur

**Instructions:**
1. Create function: `apply_gaussian_blur(image, kernel_size)`
2. Understand Gaussian kernel:
   - What does kernel_size control?
   - What's the effect of sigma?
   - Why Gaussian vs other filters?
3. Experiment with kernel sizes (3, 5, 7, 9)
4. Visualize effect on edges

**Key Concept:** Noise-Edge Tradeoff
- Too much blur → lose lane edges
- Too little blur → noisy edges
- Find optimal balance

**Deliverable:** Comparison visualization

---

#### Step 2.3: Region of Interest (ROI)
**Task:** Mask irrelevant image regions

**Instructions:**
1. Create function: `define_roi(image, vertices)`
2. Define trapezoid/polygon vertices:
   ```
   Bottom-left ──────── Bottom-right
      │                    │
      │   (road region)    │
      │                    │
   Top-left ──────────── Top-right
   ```
3. Create binary mask
4. Apply mask to keep only ROI

**Key Concept:** Why ROI?
- Lanes appear in predictable regions
- Reduce false positives (trees, signs, etc.)
- Improve computational efficiency

**Challenge:** How to set vertices automatically?
- Image size dependent
- Camera mounting dependent
- Think about generalization

**Deliverable:** Function with configurable ROI

---

### **PHASE 3: Edge Detection (2-3 hours)**

#### Step 3.1: Understand Gradients
**Task:** Implement Sobel edge detection

**Instructions:**
1. Study Sobel operator:
   ```
   Gx = [-1 0 1]    Gy = [-1 -2 -1]
        [-2 0 2]         [ 0  0  0]
        [-1 0 1]         [ 1  2  1]
   ```
2. Create function: `compute_gradient(image)`
3. Compute Gx (horizontal gradients)
4. Compute Gy (vertical gradients)
5. Compute magnitude: sqrt(Gx² + Gy²)
6. Compute direction: arctan(Gy/Gx)

**Key Concept:** Gradients = Edge Strength
- Large gradient = strong edge
- Gradient direction = edge orientation
- Lanes have strong vertical gradients

**Visualization:** Show Gx, Gy, magnitude, direction

---

#### Step 3.2: Canny Edge Detection
**Task:** Implement full Canny algorithm

**Instructions:**
1. Create function: `detect_edges_canny(image, low_threshold, high_threshold)`
2. Understand Canny stages:
   - Gradient computation (you did this!)
   - Non-maximum suppression (thin edges)
   - Double thresholding (strong/weak edges)
   - Edge tracking by hysteresis
3. Use OpenCV's `Canny()`
4. Experiment with thresholds:
   - Low threshold: 50
   - High threshold: 150
   - Adjust based on results

**Key Concept:** Why Canny?
- Optimal edge detector (3 criteria)
- Reduces false edges
- Produces thin, connected edges

**Challenge:** Automatic threshold selection
- Can you compute thresholds from image statistics?
- Read about Otsu's method

**Deliverable:** Edge detection with tuned parameters

---

### **PHASE 4: Line Detection (3-4 hours)**

#### Step 4.1: Hough Transform Theory
**Task:** Understand line representation

**Instructions:**
1. Study two representations:
   - Cartesian: y = mx + b
   - Polar: ρ = x cos(θ) + y sin(θ)
2. Understand Hough space:
   - Each point → sinusoidal curve in Hough space
   - Intersecting curves → collinear points
3. Visualize transformation

**Key Concept:** Parameter Space Voting
- Edge pixels "vote" for possible lines
- Lines with many votes are detected
- Threshold controls sensitivity

**Exercise:** Draw Hough space for 3 points

---

#### Step 4.2: Implement Hough Transform
**Task:** Detect lines in edge image

**Instructions:**
1. Create function: `detect_lines_hough(edges, rho, theta, threshold, min_line_len, max_line_gap)`
2. Use OpenCV's `HoughLinesP()` (probabilistic version)
3. Understand parameters:
   - `rho`: Distance resolution (pixels)
   - `theta`: Angle resolution (radians)
   - `threshold`: Minimum votes
   - `min_line_len`: Minimum line length
   - `max_line_gap`: Maximum gap between segments
4. Experiment with values:
   - Start with: rho=1, theta=π/180, threshold=50
   - Adjust based on results

**Challenge:** Parameter sensitivity
- Too low threshold → too many lines
- Too high threshold → miss lanes
- Document your tuning process

**Deliverable:** Detected lines overlay on original image

---

#### Step 4.3: Lane Line Filtering
**Task:** Separate left and right lanes

**Instructions:**
1. Create function: `separate_lanes(lines, image_width)`
2. For each detected line:
   - Compute slope: m = (y2 - y1) / (x2 - x1)
   - If slope < -0.5 → left lane (negative slope)
   - If slope > 0.5 → right lane (positive slope)
   - Discard nearly horizontal lines (|slope| < 0.5)
3. Group lines by position:
   - Lines on left half of image → left lane
   - Lines on right half → right lane

**Key Concept:** Geometric Constraints
- Lanes have specific slopes
- Lanes are on specific sides
- Use domain knowledge to filter

**Deliverable:** Separated left/right lane segments

---

#### Step 4.4: Lane Line Averaging
**Task:** Combine segments into single lane lines

**Instructions:**
1. Create function: `average_lane_lines(lines, image_height)`
2. For left lane:
   - Collect all (x, y) points from segments
   - Fit line using least squares: `np.polyfit(y, x, 1)`
   - Why fit x vs y? (avoid vertical line issues)
3. Repeat for right lane
4. Extrapolate to full image height:
   - From bottom of image (y = height)
   - To top of ROI (y = height * 0.6)

**Key Concept:** Robust Estimation
- Multiple segments → single stable line
- Outliers are averaged out
- Extrapolation fills gaps

**Challenge:** Weighted averaging
- Can you weight longer segments more?
- Can you reject outlier segments?

**Deliverable:** Two clean lane lines

---

### **PHASE 5: Visualization & Integration (1-2 hours)**

#### Step 5.1: Overlay Visualization
**Task:** Draw lane lines on original image

**Instructions:**
1. Create function: `draw_lanes(image, lines, color=(255,0,0), thickness=10)`
2. Create transparent overlay:
   - Use OpenCV's `addWeighted()` for transparency
3. Add lane region fill:
   - Create polygon between left and right lines
   - Fill with semi-transparent color
4. Add confidence indicators:
   - Text showing number of detected segments
   - Line thickness based on confidence

**Deliverable:** Publication-quality visualizations

---

#### Step 5.2: Full Pipeline Integration
**Task:** Combine all steps

**Instructions:**
1. Create `main.py` with function:
   ```python
   def detect_lanes(image_path):
       # 1. Load image
       # 2. Grayscale conversion
       # 3. Gaussian blur
       # 4. ROI mask
       # 5. Canny edge detection
       # 6. Hough line detection
       # 7. Lane separation
       # 8. Lane averaging
       # 9. Visualization
       return result_image
   ```
2. Add error handling
3. Add logging
4. Make parameters configurable (config file or argparse)

**Deliverable:** Single-function lane detector

---

### **PHASE 6: Testing & Analysis (1-2 hours)**

#### Step 6.1: Unit Tests
**Task:** Write tests for each component

**Instructions:**
1. Test preprocessing:
   - Grayscale output shape correct
   - Blur reduces noise (measure variance)
   - ROI mask has correct shape
2. Test edge detection:
   - Edges are binary
   - Edge density reasonable
3. Test line detection:
   - Lines have correct format
   - Slopes in expected range

**Deliverable:** Passing test suite

---

#### Step 6.2: Performance Analysis
**Task:** Evaluate on test set

**Instructions:**
1. Run pipeline on all test images
2. For each result, analyze:
   - Success/failure?
   - If failure, why?
   - What conditions cause failure?
3. Create comparison table:
   | Image | Lighting | Conditions | Success | Notes |
   |-------|----------|------------|---------|-------|
4. Document failure modes

**Key Insight:** When Classical CV Fails
- Shadows/lighting changes
- Curved lanes
- Faded markings
- Occlusions
- Complex scenes

**Deliverable:** Analysis report in docs/

---

### **PHASE 7: Extensions (Optional, +2-4 hours)**

#### Extension 1: Curved Lane Detection
**Task:** Modify to detect curved lanes

**Hint:**
- Use polynomial fitting instead of linear
- `np.polyfit(y, x, 2)` for quadratic
- Adjust visualization

---

#### Extension 2: Video Processing
**Task:** Apply to video stream

**Hint:**
- Read video frame-by-frame
- Add temporal smoothing between frames
- Track lane position over time

---

#### Extension 3: Automatic Parameter Tuning
**Task:** Find optimal parameters automatically

**Hint:**
- Grid search over parameters
- Define "good detection" metric
- Use validation set

---

## 📊 Evaluation Rubric

### **Code Quality (30%)**
- [ ] Clean, readable code
- [ ] Proper function decomposition
- [ ] Meaningful variable names
- [ ] Comments explaining "why", not "what"
- [ ] Follows PEP 8 style guide

### **Documentation (25%)**
- [ ] Clear README with usage instructions
- [ ] Methodology document explaining approach
- [ ] Inline comments for complex logic
- [ ] Results documented with images
- [ ] Known limitations discussed

### **Functionality (25%)**
- [ ] Correct grayscale conversion
- [ ] Effective edge detection
- [ ] Accurate line detection
- [ ] Proper lane separation
- [ ] Clean visualization

### **Understanding (20%)**
- [ ] Can explain each step's purpose
- [ ] Understands parameter effects
- [ ] Recognizes failure modes
- [ ] Articulates limitations
- [ ] Proposes improvements

---

## 🎓 Key Takeaways

After completing this project, you should understand:

1. **Computer Vision Pipeline:**
   - Preprocessing → Feature Detection → Analysis → Visualization
   - Each step builds on previous

2. **Classical CV Strengths:**
   - Fast, efficient
   - Interpretable
   - No training data needed
   - Good for constrained scenarios

3. **Classical CV Limitations:**
   - Brittle to variations
   - Manual parameter tuning
   - Poor generalization
   - Fails in complex scenes

4. **Why Deep Learning?**
   - Learns features automatically
   - Robust to variations
   - Generalizes better
   - Handles complexity

---

## 📚 Recommended Reading

**Before Starting:**
- OpenCV Tutorials: Image Processing Basics
- Szeliski Book: Chapter 4 (Feature Detection)

**While Working:**
- Canny Edge Detection paper (1986)
- Hough Transform tutorial (online)

**After Completing:**
- Compare with modern methods
- Read 2D lane detection papers

---

## 💡 Debugging Tips

**No edges detected?**
- Check Canny thresholds (too high?)
- Verify grayscale conversion
- Visualize each step

**Too many false positives?**
- Tighten Hough threshold
- Adjust slope filtering
- Improve ROI mask

**Lines unstable/jittery?**
- Add temporal smoothing
- Increase line length threshold
- Better segment averaging

**Pipeline slow?**
- Reduce image resolution
- Optimize ROI size
- Profile code (cProfile)

---

## 🚀 Next Steps

After completing Project 1:
1. Document lessons learned
2. Commit code to GitHub
3. Write summary in README
4. Move to Project 2: Perspective Transformation
5. Think about how to improve this approach

---

## 📝 Submission Checklist

Before moving to next project:
- [ ] All code implemented and tested
- [ ] Test suite passes
- [ ] Results documented with images
- [ ] README complete
- [ ] Methodology doc written
- [ ] Known limitations identified
- [ ] GitHub repo ready
- [ ] Personal understanding solid

---

**Remember:** The goal is understanding, not just working code. Take time to experiment, visualize, and reflect. Good luck! 🎓
