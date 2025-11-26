import cv2


def detect_edge_canny(image, low_threshold=50, high_threshold=150):
    """Detect edges using Canny edge detector."""
    return cv2.Canny(image, low_threshold, high_threshold)
