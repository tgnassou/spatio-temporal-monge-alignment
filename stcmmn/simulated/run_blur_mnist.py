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
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output
# %%
from skorch import NeuralNetClassifier

model = NeuralNetClassifier(
    Net,
    max_epochs=10,
    lr=0.1,
    batch_size=1000,
    device="cuda",
    optimizer=torch.optim.Adadelta,
    criterion=nn.CrossEntropyLoss,
)


# %%
X_train = np.concatenate(X[:9]).astype(np.float32).reshape(-1, 1, 28, 28)
y_train = np.repeat(train_labels_numpy, 9)
# %%
model.fit(train_data_numpy, train_labels_numpy)
# %%
print(model.score(np.concatenate(X[10:]), np.repeat(train_labels_numpy, 5)))

