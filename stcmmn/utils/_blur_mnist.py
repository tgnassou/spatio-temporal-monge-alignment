import numpy as np
from scipy.signal import convolve2d


def create_1d_gaussian(length, sigma):
    """
    Create a 1D Gaussian profile.

    Parameters:
        length (int): Length of the Gaussian profile.
        sigma (float): Standard deviation of the Gaussian profile.

    Returns:
        numpy.ndarray: 1D Gaussian profile.
    """
    x = np.arange(length)
    gaussian = np.exp(-((x - length // 2) ** 2) / (2 * sigma**2))
    return gaussian / np.sum(gaussian)


def create_2d_gaussian_filter(height, width, angle_deg, sigma):
    """
    Create a 2D image representing a Gaussian filter in one direction.

    Parameters:
        height (int): Height of the image.
        width (int): Width of the image.
        angle_deg (float): Angle of the line in degrees.
        sigma (float): Standard deviation of the Gaussian profile
        along the line.

    Returns:
        numpy.ndarray: 2D image representing the Gaussian filter.
    """
    # Initialize the image with zeros
    image = np.zeros((height, width))

    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2

    # Convert angle to radians
    if angle_deg > 90:
        angle_rad = np.radians(angle_deg - 90)
    else:
        angle_rad = np.radians(angle_deg)

    # Compute unit vector components
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Calculate vertical shift to transform angle to 0-90 degrees

    # Calculate the coordinates along the line
    for extent in range(-height // 2, height // 2 + 1):
        y = int(center_y + extent * dy)
        x = int(center_x + extent * dx)

        if 0 <= x < width and 0 <= y < height:
            image[y, x] = np.exp(-(extent ** 2) / (2 * sigma ** 2))
    if angle_deg > 90:
        image = np.rot90(image)
    return image


def apply_convolution(image, gaussian_filter_image):
    blurred_image = convolve2d(
        image[0], gaussian_filter_image, mode="same", boundary="wrap"
    )
    return blurred_image


def welch_method(image, window_size, overlap_ratio):
    """
    Estimate the power spectral density of an image using the Welch method.

    Parameters:
        image (numpy.ndarray): Input image.
        window_size (int): Size of the window for each segment.
        overlap_ratio (float): Overlap ratio between consecutive segments.

    Returns:
        numpy.ndarray: Estimated power spectral density.
    """
    # Calculate overlap size
    overlap_size = int(window_size * overlap_ratio)

    # Initialize list to store PSDs of each patch
    psd_patches = []

    # Loop over the image, extract patches, and compute PSDs
    for i in range(0, image.shape[0] - window_size + 1, overlap_size):
        for j in range(0, image.shape[1] - window_size + 1, overlap_size):
            patch = image[i:i + window_size, j:j + window_size]

            # Compute FFT of the segment and its squared magnitude
            fft = np.fft.fft(patch.flatten(), axis=0)
            psd = np.abs(fft) ** 2

            # Store the periodogram of the segment
            psd_patches.append(psd)

    # Average the periodograms of all segments
    # Average the PSDs of all patches
    average_psd = np.mean(psd_patches, axis=0)

    return average_psd


def compute_psd(image):
    fft = np.fft.fft2(image)
    psd = np.abs(fft) ** 2
    return psd
