import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier
from torch import nn
import torch
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from stcmmn.utils import load_BCI_dataset
from stcmmn.utils import CMMN, RiemanianAlignment
from stcmmn.utils import AdaptiveShallowFBCSPNet
from stcmmn.utils import DATASET_PARAMS, compute_final_conv_length

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Sleep DA", description="Domain Adaptation on sleep EEG."
    )
    # dataset to use, when expe is inter this dataset is the one use in target
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("--concatenate", action='store_true')
    parser.add_argument("--archi", type=str, default="ShallowNet")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--filter-size", type=int, default=32)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--savedir", type=str, default="LODO")

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

n_jobs = 10

dataset_names = ["BNCI2014001", "Cho2017", "Weibo2014", "PhysionetMI"]
channels = DATASET_PARAMS["BNCI2014001"]["channels"]
fs = 128
mapping = {"right_hand": 0, "left_hand": 1}
dataset_dict = {}
for dataset in dataset_names:

    subject_id = DATASET_PARAMS[dataset]["subject_id"]
    if dataset == "PhysionetMI":
        mapping = DATASET_PARAMS[dataset]["mapping"]
    else:
        mapping = {"right_hand": 0, "left_hand": 1}

    X, y = load_BCI_dataset(
        dataset, subject_id=subject_id, n_jobs=n_jobs,
        filter=args.filter, resample=fs, channels_to_pick=channels,
        mapping=mapping,
    )

    if dataset == "PhysionetMI":
        X = [X[i][:3] for i in range(len(X))]
        y = [y[i][:3] for i in range(len(y))]

    dataset_dict[dataset] = (X, y)


n_epochs = args.n_epochs
filter_size = args.filter_size
archi = args.archi
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 256
seed = 42
for dataset_target in dataset_names:
    for method in ["raw", "spatiotemp", "riemann"]:
        results_path = (
            f"results/{args.savedir}/{archi}_{method}"
            f"_to_{dataset_target}"
            f"_filter_{args.filter}.pkl"
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        X_source = []
        y_source = []
        for dataset in dataset_names:
            if dataset != dataset_target:
                X, y = dataset_dict[dataset]
                X_ = []
                for x in X:
                    X_ += [x[i][:, :, :fs * 3] for i in range(len(x))]
                y_ = []
                for y__ in y:
                    y_ += y__
                X_source += X_
                y_source += y_
        X_all_target, y_all_target = dataset_dict[dataset_target]

        y_train = np.concatenate(np.array(y_source), axis=0)

        n_classes = np.unique(y_train).shape[0]
        n_domains = len(X_source)
        if method in ["temp", "spatiotemp"]:
            cmmn = CMMN(
                method=method,
                filter_size=filter_size,
                fs=fs,
                reg=args.reg,
                concatenate_epochs=args.concatenate
            )
            X_norm_ = cmmn.fit_transform(X_source)
        elif method == "raw":
            X_norm_ = X_source
        elif method == "riemann":
            ra = RiemanianAlignment()
            X_norm_ = ra.fit_transform(X_source)
        else:
            raise ValueError("Method not implemented")
        X_train = np.concatenate(X_norm_, axis=0)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
        )
        dataset_val = Dataset(X_val, y_val)
        n_chans, n_times = X_[0][0].shape
        if archi == "ShallowNet":
            model = ShallowFBCSPNet(
                n_chans,
                n_classes,
                n_times=n_times,
                final_conv_length='auto',
                add_log_softmax=False,
            )
        elif archi == "EEGNet":
            model = EEGNetv4(
                n_chans,
                n_classes,
                n_times=n_times,
                final_conv_length='auto',
                add_log_softmax=False,
            )
        # elif archi == "AdaptiveShallowNet":
        #     final_conv_length = compute_final_conv_length(n_times, n_times_target)
        #     model = AdaptiveShallowFBCSPNet(
        #         n_chans,
        #         n_classes,
        #         n_times=n_times,
        #         final_conv_length=final_conv_length,
        #         add_log_softmax=False,
        #     )
        model = model.to(device)

        clf = EEGClassifier(
            module=model,
            max_epochs=n_epochs,
            batch_size=batch_size,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            device=device,
            train_split=predefined_split(dataset_val),
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)
                ),
                ("early_stopping", EarlyStopping(patience=20)),
            ],
        )

        clf.fit(X_train, y_train)
        # Predict y for X_test
        n_subjects = len(X_all_target)
        for subject_test in range(n_subjects):
            X_test = X_all_target[subject_test]
            y_test = y_all_target[subject_test]
            X_test = [X_test[i][:, :, :fs * 3] for i in range(len(X_test))]
            if method in ["temp", "spatiotemp"]:
                X_norm = cmmn.transform(X_test)
            elif method == "raw":
                X_norm = X_test
            elif method == "riemann":
                X_norm = ra.transform(X_test)
            else:
                raise ValueError("Method not implemented")

            X_norm = np.concatenate(X_norm, axis=0)
            y_pred = clf.predict(X_norm)
            y_true = np.concatenate(y_test, axis=0)
            results = [{
                "acc": accuracy_score(y_true, y_pred),
                "archi": archi,
                "method": method,
                "subject_test": subject_test,
                "seed": seed,
                "filter": args.filter,
                "filter_size": filter_size,
                "dataset_target": dataset_target,
                "n_epochs": n_epochs,
                "concatenate": args.concatenate,
                "reg": args.reg,
            }]

            try:
                df_results = pd.read_pickle(results_path)
            except FileNotFoundError:
                df_results = pd.DataFrame()
            df_results = pd.concat((df_results, pd.DataFrame(results)))
            df_results.to_pickle(results_path)
