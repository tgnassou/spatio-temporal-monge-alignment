# %%
from stcmmn.utils import DATASET_PARAMS
from stcmmn.utils import load_BCI_dataset
from stcmmn.utils import CMMN
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette("colorblind", 10)
# %%
dataset = "Cho2017"

fs = DATASET_PARAMS[dataset]["fs"]
subject_id = DATASET_PARAMS[dataset]["subject_id"]
channels = DATASET_PARAMS[dataset]["channels"]
mapping = DATASET_PARAMS[dataset]["mapping"]

# %%
X_all, y_all = load_BCI_dataset(
    dataset, subject_id=[1, 2, 3], n_jobs=1,
    filter=True, channels_to_pick=channels, mapping=mapping,
    resample=100
)

# %%
reg = 0.1
filter_size = 64
X = np.concatenate(X_all)
cmmn = CMMN(
    method="spatiotemp", filter_size=filter_size, fs=fs, reg=reg, num_iter=100
)
X_transform = cmmn.fit_transform(np.concatenate(X_all))
X_concat = [np.concatenate(X[i], axis=-1) for i in range(len(X))]
H = cmmn.compute_filter(X_concat)
# H = cmmn.compute_filter(X)

barycenter = cmmn.barycenter
# %%
freqs = scipy.signal.welch(X_concat[0][0], fs=100, nperseg=filter_size)[0]
psd = [cmmn._get_csd(X_concat[i]) for i in range(len(X_concat))]
X_transform_concat = [
    np.concatenate(X_transform[i], axis=-1) for i in range(len(X_transform))
]
psd_transform = [
    cmmn._get_csd(X_transform_concat[i])
    for i in range(len(X_transform_concat))
]
# %%
figsize = (5, 5)
fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
time = np.arange(len(H[0, 0, 0])) / fs
for i in range(3):
    for j in range(3):
        axes[i, j].plot(
            time, H[0, i, j],
            color=palette[2], alpha=0.7, linewidth=0.9
        )
        axes[i, j].grid()
fig.suptitle(f"Filter (reg = {reg}, filter_size = {filter_size})")
axes[0, 0].set_ylabel("Channel 1")
axes[1, 0].set_ylabel("Channel 2")
axes[2, 0].set_ylabel("Channel 3")
axes[2, 0].set_xlabel("Time (s)")
axes[2, 1].set_xlabel("Time (s)")
axes[2, 2].set_xlabel("Time (s)")
axes[0, 0].set_title("Channel 1")
axes[0, 1].set_title("Channel 2")
axes[0, 2].set_title("Channel 3")

axes[0, 0].set_yticks([])


fig.tight_layout()

# %%
figsize = (5, 5)
fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
time = np.arange(len(H[0, 0, 0])) / fs
for i in range(3):
    for j in range(3):
        axes[i, j].plot(
            freqs, np.fft.rfftn(np.fft.fftshift(H[0, i, j])),
            color=palette[2], alpha=0.7, linewidth=0.9
        )
        axes[i, j].grid()
        # axes[i, j].set_yscale("log")
        # axes[i, j].set_ylim(1e-5, 10)
fig.suptitle(f"Filter (reg = {reg}, filter_size = {filter_size})")
axes[0, 0].set_ylabel("Channel 1")
axes[1, 0].set_ylabel("Channel 2")
axes[2, 0].set_ylabel("Channel 3")
axes[2, 0].set_xlabel("Freq. (Hz)")
axes[2, 1].set_xlabel("Freq. (Hz)")
axes[2, 2].set_xlabel("Freq. (Hz)")
axes[0, 0].set_title("Channel 1")
axes[0, 1].set_title("Channel 2")
axes[0, 2].set_title("Channel 3")

axes[0, 0].set_yticks([])


fig.tight_layout()
# %%
fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)

for k in range(len(X_concat)):
    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                freqs, psd[k][i, j],
                color=palette[0], alpha=0.2, linewidth=0.5
            )
            axes[i, j].grid()
for i in range(3):
    for j in range(3):
        axes[i, j].plot(
            freqs, barycenter[i, j],
            color="black", alpha=0.7, linewidth=1.5, linestyle="--"
        )
        axes[i, j].grid()
        # axes[i, j].set_xlim(0, 20)
        axes[i, j].set_yscale("log")
fig.suptitle(f"PSD (reg = {reg}, filter_size = {filter_size})")
axes[0, 0].set_ylabel("Channel 1")
axes[1, 0].set_ylabel("Channel 2")
axes[2, 0].set_ylabel("Channel 3")
axes[2, 0].set_xlabel("Freq. (Hz)")
axes[2, 1].set_xlabel("Freq. (Hz)")
axes[2, 2].set_xlabel("Freq. (Hz)")
axes[0, 0].set_title("Channel 1")
axes[0, 1].set_title("Channel 2")
axes[0, 2].set_title("Channel 3")

axes[0, 0].set_yticks([])

fig.tight_layout()
# %%
fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)

for k in range(len(X_concat)):
    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                freqs, psd_transform[k][i, j],
                color=palette[1], alpha=0.2, linewidth=0.5
            )
            axes[i, j].grid()
for i in range(3):
    for j in range(3):
        axes[i, j].plot(
            freqs, barycenter[i, j],
            color="black", alpha=0.7, linewidth=1.5, linestyle="--"
        )
        axes[i, j].grid()
        # axes[i, j].set_xlim(0, 20)
        axes[i, j].set_yscale("log")
fig.suptitle(f"Transformed PSD (reg = {reg}, filter_size = {filter_size})")
axes[0, 0].set_ylabel("Channel 1")
axes[1, 0].set_ylabel("Channel 2")
axes[2, 0].set_ylabel("Channel 3")
axes[2, 0].set_xlabel("Freq. (Hz)")
axes[2, 1].set_xlabel("Freq. (Hz)")
axes[2, 2].set_xlabel("Freq. (Hz)")
axes[0, 0].set_title("Channel 1")
axes[0, 1].set_title("Channel 2")
axes[0, 2].set_title("Channel 3")

axes[0, 0].set_yticks([])

fig.tight_layout()
# %%
