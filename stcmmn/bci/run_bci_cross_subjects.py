import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse

from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode import EEGClassifier
from torch import nn
import torch
from skorch.callbacks import LRScheduler

from stcmmn.utils import load_BCI_dataset
from stcmmn.utils import CMMN, RiemanianAlignment
from stcmmn.utils import AdaptiveShallowFBCSPNet
from stcmmn.utils import DATASET_PARAMS

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Sleep DA", description="Domain Adaptation on sleep EEG."
    )
    # dataset to use, when expe is inter this dataset is the one use in target
    parser.add_argument("--filter", action='store_true')
    parser.add_argument("--method", type=str, default="raw")
    parser.add_argument("--dataset", type=str, default="BNCI2014001")
    parser.add_argument("--archi", type=str, default="ShallowNet")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--filter-size", type=int, default=128)
    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

n_jobs = 10
n_subjects = None

fs = DATASET_PARAMS[args.dataset]["fs"]
subject_id = DATASET_PARAMS[args.dataset]["subject_id"]
channels = DATASET_PARAMS[args.dataset]["channels"]
mapping = DATASET_PARAMS[args.dataset]["mapping"]

X_all, y_all = load_BCI_dataset(
    args.dataset, subject_id=subject_id, n_jobs=n_jobs,
    filter=args.filter, channels_to_pick=channels,
    mapping=mapping,
)

n_subjects = len(X_all)
n_epochs = args.n_epochs
filter_size = args.filter_size
method = args.method
archi = args.archi
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
results_path = (
    f"results/cross_subjects/{archi}_{method}"
    f"_dataset_{args.dataset}"
    f"_filter_{args.filter}.pkl"
)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
for subject_test in range(len(X_all)):
    print(f"Subject {subject_test}")
    X = np.array(X_all)[np.arange(n_subjects) != subject_test]
    y = np.array(y_all)[np.arange(n_subjects) != subject_test]
    n_session = len(X)
    # Create X_train and X_test with leave one subject out

    X_ = np.concatenate(X, axis=0)
    y_train = np.concatenate(np.concatenate(y, axis=0), axis=0)

    n_classes = np.unique(y_train).shape[0]
    n_domains = len(X_)
    n_chans, n_times = X_[0][0].shape
    if method in ["temp", "spatiotemp"]:
        cmmn = CMMN(
            method=method, filter_size=filter_size, fs=fs,
        )
        X_norm_ = cmmn.fit_transform(X_)
    elif method == "raw":
        X_norm_ = X_
    elif method == "riemann":
        ra = RiemanianAlignment()
        X_norm_ = ra.fit_transform(X_)
    else:
        raise ValueError("Method not implemented")
    X_train = np.concatenate(X_norm_, axis=0)

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
    elif archi == "AdaptiveShallowNet":
        model = AdaptiveShallowFBCSPNet(
            n_chans,
            n_classes,
            n_times=n_times,
            final_conv_length=20,
            add_log_softmax=False,
        )
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
        train_split=None,
        device=device,
        callbacks=[(
            "lr_scheduler",
            LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)
        )],
    )

    clf.fit(X_train, y_train)
    # Predict y for X_test
    X_test = X_all[subject_test]
    y_test = y_all[subject_test]

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
        "dataset": args.dataset,
        "n_epochs": n_epochs,
    }]

    try:
        df_results = pd.read_pickle(results_path)
    except FileNotFoundError:
        df_results = pd.DataFrame()
    df_results = pd.concat((df_results, pd.DataFrame(results)))
    df_results.to_pickle(results_path)
