import cv2
import numpy as np
from preprocessing import define_roi

def detect_line_hough(image, rho=1, theta=np.pi / 180, threshold=100, min_line_length=50, max_line_gap=10):
    """Detect lines using Hough Transform."""
    lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """Draw lines on the image."""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def process_image_for_lines(image, vertices):
    """Process the image to detect and draw lines."""
    masked_image = define_roi(image, vertices)
    lines = detect_line_hough(masked_image)
    line_image = draw_lines(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), lines)
    return line_image


def separate_lanes(lines, img_shape):
    """Separate lines into left and right lanes based on their slope."""
    left_lines = []
    right_lines = []
    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Avoid division by zero - skip vertical lines
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append(line)
        elif slope > 0.5:
            right_lines.append(line)
        else:
            continue  # Ignore near-horizontal lines
    return left_lines, right_lines

def average_lane_line(lines, img_height):
    """Average the position of lane lines and extrapolate to the bottom of the image."""
    if len(lines) == 0:
        return None

    x_coords = []
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    # Use np.polyfit for least squares fitting (fit x as function of y)
    # This gives us the line equation: x = slope * y + intercept
    slope, intercept = np.polyfit(y_coords, x_coords, 1)

    # Calculate x coordinates for y positions
    y1 = img_height
    y2 = int(img_height * 0.6)
    x1 = int(slope * y1 + intercept)
    x2 = int(slope * y2 + intercept)

    return np.array([[x1, y1, x2, y2]])

def draw_lane_lines(image, left_line, right_line, color=(0, 0, 255), thickness=5):
    """Draw the averaged lane lines on the image."""
    line_image = np.zeros_like(image)
    if left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    if right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    return combined_image

def process_image_for_lanes(image, vertices):
    """Process the image to detect and draw lane lines."""
    masked_image = define_roi(image, vertices)
    lines = detect_line_hough(masked_image)
    left_lines, right_lines = separate_lanes(lines, image.shape)

    left_lane = average_lane_line(left_lines, image.shape[0])
    right_lane = average_lane_line(right_lines, image.shape[0])

    lane_image = draw_lane_lines(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), left_lane, right_lane)
    return lane_image

