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
from stcmmn.utils import CMMN
from stcmmn.utils import DATASET_PARAMS

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
    parser.add_argument("--dataset", type=str, default="BNCI2014001")
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--savedir", type=str, default="sensitivity_study")

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

n_jobs = 10

# Load the three datasets
dataset_names = ["BNCI2014001", "Weibo2014", "PhysionetMI"]
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

# Define source and target domains
dataset_target = args.dataset
X_source = []
y_source = []
for dataset in dataset_names:
    if dataset != dataset_target:
        X, y = dataset_dict[dataset]
        X_ = []
        for x in X:
            time_length = x[0].shape[2]
            length_to_remove = (time_length - (fs * 3)) // 2
            X_ += [
                x[i][:, :, length_to_remove:length_to_remove + (fs * 3)]
                for i in range(len(x))
            ]
        y_ = []
        for y__ in y:
            y_ += y__
        X_source += X_
        y_source += y_
X_all_target, y_all_target = dataset_dict[dataset_target]

y_train = np.concatenate(np.array(y_source), axis=0)

n_classes = np.unique(y_train).shape[0]
n_chans, n_times = X_source[0][0].shape
n_domains = len(X_source)

# Define parameters
n_epochs = args.n_epochs
archi = args.archi
lr = 0.0625 * 0.01
weight_decay = 0
batch_size = args.batch_size
seed = 42
method = "spatiotemp"
np.random.seed(seed)
torch.manual_seed(seed)
results_path = (
    f"results/{args.savedir}/{archi}_{method}"
    f"_to_{dataset_target}"
    f"_filter_{args.filter}.pkl"
)

for filter_size in [2, 4, 8, 16, 32, 64, 128]:
    for reg in [0, 0.001, 0.01,]:
        cmmn = CMMN(
            method=method,
            filter_size=filter_size,
            reg=reg,
            concatenate_epochs=args.concatenate
        )
        X_adapted = cmmn.fit_transform(X_source)
        X_train = np.concatenate(X_adapted, axis=0)

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
            train_split=None,
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)
                ),
            ],
        )

        clf.fit(X_train, y_train)

        # Predict y for X_test
        n_subjects = len(X_all_target)
        for subject_test in range(n_subjects):
            X_test = X_all_target[subject_test]
            y_test = y_all_target[subject_test]
            time_length = X_test[0].shape[2]
            length_to_remove = (time_length - (fs * 3)) // 2
            X_test = [
                X_test[i][:, :, length_to_remove:length_to_remove + (fs * 3)]
                for i in range(len(X_test))
            ]
            X_test_adapted = cmmn.transform(X_test)

            X_test_adapted = np.concatenate(X_test_adapted, axis=0)
            y_pred = clf.predict(X_test_adapted)
            y_true = np.concatenate(y_test, axis=0)

            # Save results
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
                "reg": reg,
                "batch_size": batch_size,
            }]
            try:
                df_results = pd.read_pickle(results_path)
            except FileNotFoundError:
                df_results = pd.DataFrame()
            df_results = pd.concat((df_results, pd.DataFrame(results)))
            df_results.to_pickle(results_path)
