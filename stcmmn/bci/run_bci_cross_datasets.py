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
    parser.add_argument("--method", type=str, default="raw")
    parser.add_argument("--dataset-source", type=str, default="BNCI2014001")
    parser.add_argument("--dataset-target", type=str, default="BNCI2014004")
    parser.add_argument("--archi", type=str, default="AdaptiveShallowNet")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--filter-size", type=int, default=128)
    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

n_jobs = 10
n_subjects = None

fs_source = DATASET_PARAMS[args.dataset_source]["fs"]
subject_id_source = DATASET_PARAMS[args.dataset_source]["subject_id"]
channels_source = DATASET_PARAMS[args.dataset_source]["channels"]
mapping_source = DATASET_PARAMS[args.dataset_source]["mapping"]

fs_target = DATASET_PARAMS[args.dataset_target]["fs"]
subject_id_target = DATASET_PARAMS[args.dataset_target]["subject_id"]
channels_target = DATASET_PARAMS[args.dataset_target]["channels"]
mapping_target = DATASET_PARAMS[args.dataset_target]["mapping"]

fs = 128
channels = np.intersect1d(channels_source, channels_target)
classes = np.intersect1d(
    list(mapping_source.keys()), list(mapping_target.keys())
)
# mapping = {class_: i for i, class_ in enumerate(classes)}
mapping = {"right_hand": 0, "left_hand": 1}
if args.dataset_source != "PhysionetMI":
    mapping_source = mapping
if args.dataset_target != "PhysionetMI":
    mapping_target = mapping
X_all_source, y_all_source = load_BCI_dataset(
    args.dataset_source, subject_id=subject_id_source, n_jobs=n_jobs,
    filter=args.filter, resample=fs, channels_to_pick=channels,
    mapping=mapping_source,
)
X_all_target, y_all_target = load_BCI_dataset(
    args.dataset_target, subject_id=subject_id_target, n_jobs=n_jobs,
    filter=args.filter, resample=fs, channels_to_pick=channels,
    mapping=mapping_target,
)

if args.dataset_source == "PhysionetMI":
    X_all_source = [X_all_source[i][:3] for i in range(len(X_all_source))]
    y_all_source = [y_all_source[i][:3] for i in range(len(y_all_source))]
if args.dataset_target == "PhysionetMI":
    X_all_target = [X_all_target[i][:3] for i in range(len(X_all_target))]
    y_all_target = [y_all_target[i][:3] for i in range(len(y_all_target))]

if args.archi != "AdaptiveShallowNet":
    for i in range(len(X_all_source)):
        for j in range(len(X_all_source[i])):
            X_all_source[i][j] = X_all_source[i][j][:, :, :fs * 3]
    for i in range(len(X_all_target)):
        for j in range(len(X_all_target[i])):
            X_all_target[i][j] = X_all_target[i][j][:, :, :fs * 3]

n_epochs = args.n_epochs
filter_size = args.filter_size
method = args.method
archi = args.archi
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 256
results_path = (
    f"results/cross_datasets/{archi}_{method}"
    f"_{args.dataset_source}_to_{args.dataset_target}"
    f"_filter_{args.filter}.pkl"
)

for seed in range(5):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_ = np.concatenate(np.array(X_all_source), axis=0)
    y_train = np.concatenate(np.concatenate(np.array(y_all_source), axis=0), axis=0)

    n_classes = np.unique(y_train).shape[0]
    n_domains = len(X_)
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

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
    )
    dataset_val = Dataset(X_val, y_val)
    n_chans, n_times = X_[0][0].shape
    n_times_target = X_all_target[0][0].shape[-1]
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
        final_conv_length = compute_final_conv_length(n_times, n_times_target)
        model = AdaptiveShallowFBCSPNet(
            n_chans,
            n_classes,
            n_times=n_times,
            final_conv_length=final_conv_length,
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
            "dataset_source": args.dataset_source,
            "dataset_target": args.dataset_target,
            "n_epochs": n_epochs,
        }]

        try:
            df_results = pd.read_pickle(results_path)
        except FileNotFoundError:
            df_results = pd.DataFrame()
        df_results = pd.concat((df_results, pd.DataFrame(results)))
        df_results.to_pickle(results_path)
