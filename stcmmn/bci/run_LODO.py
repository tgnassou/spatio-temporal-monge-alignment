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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--savedir", type=str, default="LODO")
    parser.add_argument("--method", type=str, default="raw")
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--filter-size", type=int, default=8)
    parser.add_argument("--num-iter", type=int, default=1)

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

n_jobs = 5

# Load the three datasets
dataset_names = [
    "BNCI2014001",
    "Weibo2014",
    "PhysionetMI",
    "Cho2017",
    "Schirrmeister2017",
]
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
        hands_index = (y[1] == 0) | (y[1] == 1)
        X = [X[i][hands_index] for i in range(len(X))]
        y = [y[i][hands_index] for i in range(len(y))]

    dataset_dict[dataset] = (X, y)

# Define source and target domains
dataset_target = args.dataset

for method in ["spatiotemp"]:
    for dataset_target in dataset_names:
        # Define source and target domains
        X_source = []
        y_source = []
        for dataset in dataset_names:
            if dataset != dataset_target:
                X, y = dataset_dict[dataset]
                time_length = X[0].shape[2]
                length_to_remove = (time_length - (fs * 3)) // 2
                X_ = [
                    X[n_sub][
                        :, :, length_to_remove:length_to_remove + (fs * 3)
                    ]
                    for n_sub in range(len(X))
                ]
                X_source += X_
                y_source += y
        X_all_target, y_all_target = dataset_dict[dataset_target]

        y_train = np.concatenate(y_source, axis=0)

        n_classes = np.unique(y_train).shape[0]
        n_chans, n_times = X_source[0][0].shape
        n_domains = len(X_source)

        # Define parameters
        n_epochs = args.n_epochs
        archi = args.archi
        # method = args.method
        filter_size = args.filter_size
        reg = args.reg
        lr = 0.0625 * 0.01
        weight_decay = 0
        batch_size = args.batch_size
        results_path = (
            f"results/{args.savedir}/{archi}_{method}"
            f"_to_{dataset_target}"
            f"_filter_{args.filter}.pkl"
        )

        if method in ["temp", "spatio", "spatiotemp"]:
            cmmn = CMMN(
                method=method,
                filter_size=filter_size,
                reg=reg,
                concatenate_epochs=args.concatenate,
                num_iter=args.num_iter,
            )
            X_adapted = cmmn.fit_transform(X_source)
        elif method == "raw":
            X_adapted = X_source
        elif method == "riemannalignment":
            ra = RiemanianAlignment()
            X_adapted = ra.fit_transform(X_source)
        else:
            raise ValueError("Method not implemented")
        X_train = np.concatenate(X_adapted, axis=0)

        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
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
                time_length = X_test.shape[2]
                length_to_remove = (time_length - (fs * 3)) // 2
                X_test = X_test[:, :, length_to_remove:length_to_remove + (fs * 3)]
                if method in ["temp", "spatio", "spatiotemp"]:
                    X_test_adapted = cmmn.transform([X_test])[0]
                elif method == "raw":
                    X_test_adapted = X_test
                elif method == "riemannalignment":
                    X_test_adapted = ra.transform(X_test)
                else:
                    raise ValueError("Method not implemented")

                # X_test_adapted = np.concatenate(X_test_adapted, axis=0)
                y_pred = clf.predict(np.array(X_test_adapted).astype(np.float32))
                # y_true = np.concatenate(y_test, axis=0)

                # Save results
                results = [{
                    "acc": accuracy_score(y_test, y_pred),
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
                    "num_iter": args.num_iter,
                }]
                try:
                    df_results = pd.read_pickle(results_path)
                except FileNotFoundError:
                    df_results = pd.DataFrame()
                df_results = pd.concat((df_results, pd.DataFrame(results)))
                df_results.to_pickle(results_path)
