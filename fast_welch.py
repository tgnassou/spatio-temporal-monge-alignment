# Example file to compute the Welch's method for a given signal

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from stcmmn.utils import CMMN

n_c = 20  # Number of sensors
n_l = 100000  # Number of time samples
f = 31  # Filter size

# Generate a test signal
X = np.random.multivariate_normal(np.zeros(n_c), np.eye(n_c), n_l).T
print(f"Shape of the input signal: {X.shape}")

# Compute the Welch's method
win = signal.windows.hann(f)
win /= np.linalg.norm(win)
stft = signal.ShortTimeFFT(
    win=win,
    hop=1,
    fs=1,
    fft_mode='onesided',
    scale_to='psd'
)
Z = stft.stft(X)
psd = np.transpose(Z, axes=(1, 0, 2)) @ np.conj(np.transpose(Z, axes=(1, 2, 0))) / n_l

# Compute the PSD from CMMN
cmmn = CMMN(filter_size=f, method="spatiotemp", reg=0)
psd_cmmn = cmmn.compute_psd(X.reshape((1, n_c, n_l)))
psd_cmmn = np.squeeze(psd_cmmn)
psd_cmmn = np.transpose(psd_cmmn, axes=(2, 0, 1))

# True PSD
true_psd = np.stack([np.eye(n_c)] * psd.shape[0])

# Errors
print(f"Error with csd: {np.linalg.norm(psd_cmmn - true_psd)}")
print(f"Error with new estimator: {np.linalg.norm(psd - true_psd)}")

# Full covariance matrix
stft.fft_mode = 'twosided'
Z = stft.stft(X)
psd = np.transpose(Z, axes=(1, 0, 2)) @ np.conj(np.transpose(Z, axes=(1, 2, 0))) / n_l
F = np.fft.fft(np.eye(f)) / np.sqrt(f)
F = np.kron(F, np.eye(n_c))
psd_matrix = np.zeros((n_c*f, n_c*f), dtype=complex)
for i in range(f):
    psd_matrix[i*n_c:(i+1)*n_c, i*n_c:(i+1)*n_c] = psd[i]
cov = F @ psd_matrix @ np.conj(F.T)
print(f"Error between full covariance matrix and new estimator: {np.linalg.norm(cov - np.eye(n_c*f))}")

# Plot the psds in a single plot
psd = np.abs(psd)
vmin = np.min(psd)
vmax = np.max(psd)
fig, ax = plt.subplots(1, f, figsize=(30, 5))
for i in range(f):
    ax[i].imshow(psd[i, :, :], vmin=vmin, vmax=vmax)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()

# Plot mean PSD
mean_psd = np.mean(psd, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(mean_psd, vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()

plt.show()
