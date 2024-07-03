# %%
import numpy as np
import scipy
from joblib import Parallel, delayed
import pandas as pd

from stcmmn.utils import (
    apply_convolution,
    compute_psd,
    create_2d_gaussian_filter,
    welch_method,
    CMMN,
)


import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from scipy.signal import convolve2d


def apply_convolution2(image, gaussian_filter_image):
    blurred_image = convolve2d(
        image, gaussian_filter_image, mode="same", boundary="wrap"
    )
    return blurred_image


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
train_data_numpy = np.load("data/MNIST/train_data.npy")
train_labels_numpy = np.load("data/MNIST/train_labels.npy")
# %%
# Example usage:
height = 28
width = 28
sigma = 2
# %%
filters = []
dirs = np.linspace(0, 180, 12, endpoint=False)
X = []
for dir in dirs:
    gaussian_filter_image = create_2d_gaussian_filter(
        height, width, dir, sigma
    )
    filters.append(gaussian_filter_image)
    blurred_images = Parallel(n_jobs=30)(
        delayed(apply_convolution)(image, gaussian_filter_image)
        for image in train_data_numpy
    )
    X.append(blurred_images)

# %%
domain_train = [0, 1, 2, 3]
# create X_train with all index except i
X_train = np.concatenate([X[i] for i in domain_train], axis=0)
X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
y_train = np.concatenate(
    [train_labels_numpy for _ in range(len(domain_train))]
)
# %%

results = []
max_epochs = 200
for seed in range(10):
    torch.manual_seed(seed)
    model = NeuralNetClassifier(
        Net,
        max_epochs=max_epochs,
        lr=1,
        batch_size=1000,
        device="cuda",
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adadelta,
        criterion=nn.CrossEntropyLoss,
        callbacks=[
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=10,
                    load_best=True
                ),
            )
        ],
        train_split=ValidSplit(cv=0.2, random_state=seed, stratified=True),

    )

    model.fit(
        X_train,
        y_train,
    )
    for i in range(len(X)):
        X_test = np.array(X[i])
        X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
        y_test = train_labels_numpy
        acc = model.score(X_test, y_test)
        results.append({
            "dir": dirs[i],
            "method": "No align.",
            "domain": i,
            "accuracy": acc,
            "seed": seed
        })
# %%
psd_domain = []

for domain in X:
    psd_welch = []
    psd = []
    for image in domain:
        estimated_psd = compute_psd(image)
        psd.append(estimated_psd)
    psd_domain.append(np.mean(psd, axis=0))

# psd_domain_welch = []

# for domain in X:
#     psd_welch = []
#     for image in domain:
#         estimated_psd = welch_method(image, window_size=12, overlap_ratio=0.3)
#         psd_welch.append(estimated_psd)
#     psd_domain_welch.append(np.mean(psd_welch, axis=0))

# %%
psd_train = [psd_domain[i] for i in domain_train]
psd_bary = np.mean(np.sqrt(psd_train), axis=0) ** 2

X_align = []
for j in range(len(X)):

    D = np.sqrt(psd_bary) / np.sqrt(psd_domain[j])
    H = scipy.fft.ifft2(D)
    H = np.real(H).reshape(28, 28)
    H = np.fft.fftshift(H, axes=(0, 1))

    aligned_images = Parallel(n_jobs=30)(
        delayed(apply_convolution2)(image, H)
        for image in X[j]
    )

    X_align.append(aligned_images)

X_train_align = np.concatenate([X_align[i] for i in domain_train], axis=0)
X_train_align = X_train_align.reshape(-1, 1, 28, 28).astype(np.float32)

# %%
results_align = []
for seed in range(10):
    torch.manual_seed(seed)
    model = NeuralNetClassifier(
        Net,
        max_epochs=max_epochs,
        lr=1,
        batch_size=1000,
        device="cuda",
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adadelta,
        criterion=nn.CrossEntropyLoss,
        train_split=ValidSplit(cv=0.2, random_state=seed, stratified=True),
        callbacks=[
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=10,
                    load_best=True
                ),
            )
        ],
    )

    model.fit(
        X_train_align,
        y_train,
    )
    for i in range(len(X)):
        X_test = np.array(X_align[i])
        X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
        y_test = train_labels_numpy
        acc = model.score(X_test, y_test)
        results_align.append({
            "dir": dirs[i],
            "method": "MA",
            "domain": i,
            "accuracy": acc,
            "seed": seed
        })

df_results = pd.DataFrame(results + results_align)
df_results.to_pickle("data/MNIST/results_one_sided_small_sigma.pkl")
# %%

# cmmn = CMMN(filter_size=1, method="spatio")
# X_train = [X[i] for i in domain_train]
# X_train = np.array(X_train).reshape(len(domain_train), -1, height * width)
# X_train_align = cmmn.fit_transform(X_train)

# X_train_align = X_train_align.reshape(-1, 1, 28, 28).astype(np.float32)

# %%
# results_align_spatial = []
# for seed in range(10):
#     torch.manual_seed(seed)
#     model = NeuralNetClassifier(
#         Net,
#         max_epochs=max_epochs,
#         lr=1,
#         batch_size=1000,
#         device="cuda",
#         iterator_train__shuffle=True,
#         optimizer=torch.optim.Adadelta,
#         criterion=nn.CrossEntropyLoss,
#         train_split=ValidSplit(cv=0.2, random_state=seed, stratified=True),
#         callbacks=[
#             (
#                 "early_stopping",
#                 EarlyStopping(
#                     monitor="valid_loss",
#                     patience=10,
#                     load_best=True
#                 ),
#             )
#         ],
#     )

#     model.fit(
#         X_train_align,
#         y_train,
#     )
#     for i in range(len(X)):
#         X_test = np.array(X_align[i]).reshape()
#         X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
#         y_test = train_labels_numpy
#         acc = model.score(X_test, y_test)
#         results_align_spatial.append({
#             "dir": dirs[i],
#             "method": "MA",
#             "domain": i,
#             "accuracy": acc,
#             "seed": seed
#         })

# # %%

