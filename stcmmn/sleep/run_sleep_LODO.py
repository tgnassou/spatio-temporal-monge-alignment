# %%
import numpy as np

from braindecode.models import SleepStagerChambon2018
from braindecode import EEGClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from stcmmn.utils import load_sleep_dataset
from stcmmn.utils import CMMN, RiemanianAlignment

import torch
from torch import nn

import pandas as pd

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

# %%
module_name = "chambon"
max_epochs = 150
batch_size = 128
patience = 15
filter_size = 2048
n_jobs = 30
# %%
for method in ["raw", "temp", "spatio", "riemann", "spatiotemp"]:

    results_path = (
        f"results/LODO_final/results_LODO_{method}_{module_name}_"
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
                        X_, y_, subject_ids_, test_size=valid_size
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
            if method in ["spatio", "spatiotemp", "temp"]:
                # compute barycenter and normalize data
                cmmn = CMMN(
                    method=method,
                    filter_size=filter_size,
                    reg=1e-3,
                    concatenate_epochs=True,
                    n_jobs=n_jobs,
                    num_iter=10
                )
                X_train = cmmn.fit_transform(X_train)
                X_val = cmmn.transform(X_val)
                X_target = cmmn.transform(X_target)
            elif method == "riemann":
                ra = RiemanianAlignment(non_homogeneous=False)
                X_train = cmmn.fit_transform(X_train)
                X_val = cmmn.transform(X_val)
                X_target = cmmn.transform(X_target)
            elif method == "raw":
                pass
            else:
                raise ValueError(f"Unknown method {method}")
            valid_dataset = Dataset(
                np.concatenate(X_val, axis=0), np.concatenate(y_val, axis=0)
            )
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(np.concatenate(y_train)),
                y=np.concatenate(y_train)
            )
            module = SleepStagerChambon2018(
                n_chans=n_channels, n_outputs=n_classes, sfreq=100
            )
            clf = EEGClassifier(
                module=module,
                max_epochs=max_epochs,
                batch_size=batch_size,
                criterion=nn.CrossEntropyLoss(
                    weight=torch.Tensor(class_weights).to(device)
                ),
                optimizer=torch.optim.Adam,
                iterator_train__shuffle=True,
                optimizer__lr=1e-3,
                device=device,
                train_split=predefined_split(valid_dataset),
                callbacks=[
                    (
                        "early_stopping",
                        EarlyStopping(
                            monitor="valid_loss",
                            patience=patience,
                            load_best=True
                        ),
                    )
                ],
            )
            clf.fit(
                np.concatenate(X_train, axis=0),
                np.concatenate(y_train, axis=0)
            )
            n_target = len(X_target)

            results = []
            for n_subj in range(n_target):
                X_t = X_target[n_subj]
                y_t = y_target[n_subj]
                subject = subject_ids_target[n_subj]
                y_pred = clf.predict(X_t)
                results.append(
                    {
                        "method": method,
                        "module": module_name,
                        "filter_size": filter_size,
                        "subject": int(subject),
                        "seed": seed,
                        "dataset_t": dataset_target,
                        "y_target": y_t,
                        "y_pred": y_pred,
                        "reg": 0,
                    }
                )
            try:
                df_results = pd.read_pickle(results_path)
            except FileNotFoundError:
                df_results = pd.DataFrame()
            df_results = pd.concat((df_results, pd.DataFrame(results)))
            df_results.to_pickle(results_path)
