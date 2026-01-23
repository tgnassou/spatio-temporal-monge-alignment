# %%
import numpy as np

from braindecode.models import SleepStagerChambon2018
from braindecode import EEGClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from monge_alignment.utils import load_sleep_dataset
from monge_alignment.utils import MongeAlignment, RiemanianAlignment

import torch
from torch import nn

import pandas as pd
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
dataset_names = [
    "ABC",
    "CHAT",
    "HOMEPAP",
    "MASS",
]

data_dict = {}
# %%
n_subject = 100
for dataset_name in dataset_names:
    X_, y_, subject_ids_ = load_sleep_dataset(
        n_subjects=n_subject,
        dataset_name=dataset_name,
        scaler="sample",
    )
    data_dict[dataset_name] = [X_, y_, subject_ids_]
    del X_, y_, subject_ids_

# %%
module_name = "chambon"
max_epochs = 150
batch_size = 128
patience = 15
filter_size = 256
n_jobs = 30
num_iter = 1
# %%
for method in ["spatio", "spatiotemp", "temp", "riemann",]:
    results_path = (
        f"results/LODO_time/results_LODO_{method}_"
        f"{len(dataset_names)}_dataset_with_{n_subject}_subjects.pkl"
    )

    for seed in range(10):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        rng = check_random_state(seed)
        for dataset_target in dataset_names:
            X_target, y_target, subject_ids_target = data_dict[dataset_target]
            X_train, X_val, y_train, y_val, subjects_train, subjects_val = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            domain_train, domain_val = [], []
            for dataset_source in dataset_names:
                if dataset_source != dataset_target:
                    X_, y_, subject_ids_ = data_dict[dataset_source]
                    valid_size = 0.2
                    (
                        X_train_,
                        X_val_,
                        y_train_,
                        y_val_,
                        subjects_train_,
                        subjects_val_,
                    ) = train_test_split(
                        X_, y_, subject_ids_,
                        test_size=valid_size, random_state=rng
                    )

                    X_train += X_train_
                    X_val += X_val_
                    y_train += y_train_
                    y_val += y_val_
                    subjects_train += subjects_train_
                    subjects_val += subjects_val_
                    domain_train += [dataset_source] * len(X_train_)
                    domain_val += [dataset_source] * len(X_val_)
                    print(
                        f"Dataset {dataset_source}: {len(X_train_)}"
                        f" train, {len(X_val_)} val"
                    )
            _, n_channels, n_time = X_train[0].shape
            n_classes = len(np.unique(y_train[0]))
            time_init = time.time()
            if method in ["spatio", "spatiotemp", "temp"]:
                # compute barycenter and normalize data
                ma = MongeAlignment(
                    method=method,
                    filter_size=filter_size,
                    reg=1e-3,
                    concatenate_epochs=True,
                    n_jobs=n_jobs,
                    num_iter=num_iter
                )
                X_train = ma.fit_transform(X_train)
                X_val = ma.transform(X_val)
                X_target = ma.transform(X_target)
                del ma
            elif method == "riemann":
                ra = RiemanianAlignment(non_homogeneous=False)
                X_train = ra.fit_transform(X_train)
                X_val = ra.transform(X_val)
                X_target = ra.transform(X_target)
            elif method == "raw":
                pass
            else:
                raise ValueError(f"Unknown method {method}")
            time_end = time.time()
            results = [{
                    "method": method,
                    "seed": seed,
                    "dataset_t": dataset_target,
                    "time_alignment": time_end - time_init,
                }]
            try:
                df_results = pd.read_pickle(results_path)
            except FileNotFoundError:
                df_results = pd.DataFrame()
            df_results = pd.concat((df_results, pd.DataFrame(results)))
            df_results.to_pickle(results_path)

# %%
