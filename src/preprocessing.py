import cv2
import numpy as np
import matplotlib.pyplot as plt


def grayscale(img):
    """Convert the image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




def apply_gaussian_blur(img, kernel_size= 5):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    """Apply Bilateral filter to the image."""
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

def display_image(title, image):
    """Display an image using matplotlib."""
    plt.figure(figsize=(20, 10))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_images_grid(images, titles, cols=6):
    """Display multiple images in a grid layout with specified columns per row."""
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))

    # Flatten axes array for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if idx < n_images:
            image = images[idx]
            title = titles[idx]

            if len(image.shape) == 2:  # Grayscale image
                ax.imshow(image, cmap='gray')
            else:  # Color image
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        else:
            # Hide unused subplots
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def define_roi(img, vertices):
    """Define Region of Interest (ROI) in the image."""
    # Create an empty mask (same HxW as the image)
    # Single channel, uint8: 0 = black (reject), 255 = white (keep)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Fill the polygon defined by vertices with white (255)
    cv2.fillPoly(mask, vertices, 255)

    # Apply the mask to the image using bitwise AND
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    return masked_image


def compute_gradient_image(image):
    """Compute the gradient magnitude and direction using Sobel operator."""
    # Compute gradients along the X and Y axis
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute gradient magnitude and direction
    magnitude = cv2.magnitude(grad_x, grad_y)
    direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    return magnitude, direction
