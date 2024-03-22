# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from joblib import Parallel, delayed

from stcmmn.utils import (
    apply_convolution,
    compute_psd,
    create_2d_gaussian_filter,
)

# %%
train_data_numpy = np.load("data/MNIST/train_data.npy")
train_labels_numpy = np.load("data/MNIST/train_labels.npy")
# %%
# Example usage:
height = 28
width = 28
sigma = 2
# %%
filters = []
dirs = np.linspace(0, 180, 10, endpoint=False)

for dir in dirs:
    domain = []
    gaussian_filter_image = create_2d_gaussian_filter(height, width, dir, sigma)
    filters.append(gaussian_filter_image)
# %%

for i in range(len(filters)):
    plt.figure()
    plt.imshow(filters[i], cmap="viridis")

# %%

X = []
dirs = np.linspace(0, 180, 10, endpoint=False)
for dir in dirs:
    gaussian_filter_image = create_2d_gaussian_filter(height, width, dir, sigma)
    blurred_images = Parallel(n_jobs=30)(
        delayed(apply_convolution)(image, gaussian_filter_image)
        for image in train_data_numpy
    )
    X.append(blurred_images)

# %%
psd_domain_welch = []
psd_domain = []

for domain in X:
    psd_welch = []
    psd = []
    for image in domain:
        estimated_psd = compute_psd(image)
        psd.append(estimated_psd)
    psd_domain.append(np.mean(psd, axis=0))

# %%
psd_bary = np.mean(np.sqrt(psd_domain[:9]), axis=0) ** 2

# %%
psd_base = []
for image in train_data_numpy:
    psd_base.append(compute_psd(image[0]))
psd_base = np.mean(psd_base, axis=0)
# %%

D = np.sqrt(psd_bary) / np.sqrt(psd_domain[9])
H = scipy.fft.ifft2(D)
H = np.real(H).reshape(28, 28)
H = np.fft.fftshift(H, axes=(0, 1))

# %%

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
            print(patch.shape)
            # Compute FFT of the segment and its squared magnitude
            fft = np.fft.fft(patch.flatten(), axis=0)
            psd = np.abs(fft) ** 2

            # Store the periodogram of the segment
            psd_patches.append(psd)

    # Average the periodograms of all segments
    # Average the PSDs of all patches
    average_psd = np.mean(psd_patches, axis=0)

    return average_psd
# %%
n_rows = 4
n_cols = 4
domain_to_plot = [0, 4, 7]
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
for i in range(n_rows):
    for j in range(n_cols):
        if i == 0:
            if j == 0:
                ax = axes[i, j]
                ax.axis("off")
            else:
                ax = axes[i, j]
                ax.imshow(train_data_numpy[j][0], cmap="viridis")
                ax.axis("off")
        else:
            if j == 0:
                ax = axes[i, j]
                ax.imshow(filters[domain_to_plot[i-1]], cmap="viridis")
            else:
                ax = axes[i, j]
                ax.imshow(X[domain_to_plot[i-1]][j], cmap="viridis")
                ax.axis("off")
        
# %%
