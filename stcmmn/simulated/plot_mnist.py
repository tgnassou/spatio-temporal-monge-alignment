# %%
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy

# Define transformations to be applied to the dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values
    ]
)

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)


# %%

# Convert training dataset to NumPy arrays
train_data_numpy = []
train_labels_numpy = []

for data, label in train_dataset:
    train_data_numpy.append(data.numpy())
    train_labels_numpy.append(label)

train_data_numpy = np.array(train_data_numpy)
train_labels_numpy = np.array(train_labels_numpy)

# Verify the shape of the arrays
print("Shape of train data array:", train_data_numpy.shape)
print("Shape of train labels array:", train_labels_numpy.shape)


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
        sigma (float): Standard deviation of the Gaussian profile along the line.

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


# %%
# Example usage:
height = 28
width = 28
angle_deg = 135
sigma = 3
gaussian_filter_image = create_2d_gaussian_filter(height, width, angle_deg, sigma)

# Plot the image4
plt.imshow(gaussian_filter_image, cmap="viridis")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %%
X = []
for dir in [0, 45, 90, 135]:
    domain = []
    gaussian_filter_image = create_2d_gaussian_filter(height, width, dir, sigma)
    for image in train_data_numpy:
        blurred_image = convolve2d(
            image[0], gaussian_filter_image, mode="same", boundary="wrap"
        )
        domain.append(blurred_image)
    X.append(domain)
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
            patch = image[i : i + window_size, j : j + window_size]

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
    fft = np.fft.fft(image.flatten(), axis=0)
    psd = np.abs(fft) ** 2
    return psd
# Example usage:
# Assuming 'image' is your input image as a numpy array
# Set window size and overlap ratio
window_size = 10
overlap_ratio = 0.5



# %%
psd_domain_welch = []
psd_domain = []

for domain in X:
    psd_welch = []
    psd = []
    for image in domain:
        estimated_psd_welch = welch_method(image, window_size, overlap_ratio)
        estimated_psd = compute_psd(image)
        psd.append(estimated_psd)
        psd_welch.append(estimated_psd_welch)
    psd_domain.append(np.mean(psd, axis=0))
    psd_domain_welch.append(np.mean(psd_welch, axis=0))

# %%
psd_bary = np.sum(np.sqrt(psd_domain), axis=0) ** 2

# %%
psd_base = []
for image in train_data_numpy:
    psd_base.append(compute_psd(image[0]))
psd_base = np.mean(psd_base, axis=0)
# %%

D = np.sqrt(psd_bary) / np.sqrt(psd_base)
H = scipy.fft.ifft(D,)
H = np.fft.fftshift(H)
H = np.real(H).reshape(28, 28)
H = np.fft.fftshift(H, axes=1)

# %%
filters = []
for dir in [0, 45, 90, 135]:
    domain = []
    gaussian_filter_image = create_2d_gaussian_filter(height, width, dir, 3)
    filters.append(gaussian_filter_image)
# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axs):
    ax.imshow(filters[i], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
# %%
