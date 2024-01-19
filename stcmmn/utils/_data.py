from mne_bids import BIDSPath, read_raw_bids
import mne

from tqdm import tqdm
import pandas as pd
import numpy as np

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from numpy import multiply
from braindecode.preprocessing import create_windows_from_events

from pathlib import Path

DATA_PATH = Path("/storage/store3/derivatives/")

mne.set_log_level("warning")


def extract_epochs(raw, eog=False, emg=False, chunk_duration=30.0):
    """Extract non-overlapping epochs from raw data.
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.
    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )

    # create a new event_id that unifies stages 3 and 4
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }

    tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=eog, emg=emg)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        picks=picks,
        preload=True,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )

    return epochs.get_data(), epochs.events[:, 2] - 1


def apply_scaler(data, method="sample"):
    if method == "sample":
        data -= np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True)
        std[std == 0] = 1
        data /= std
    elif method == "overall":
        data -= data.mean(axis=(0, 2), keepdims=True)
        data /= data.std(axis=(0, 2), keepdims=True)
    else:
        raise ValueError("Unknown scaler method")
    return data


def read_raw_bids_with_preprocessing(
    bids_path, scaler, eog=False, emg=False, to_microvolt=True
):
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    data, event = extract_epochs(raw, eog, emg)
    if to_microvolt:
        data *= 1e6
    if scaler:
        data = apply_scaler(data, method=scaler)

    return data.astype("float32"), event.astype("int64")


def load_data(
    n_subjects,
    data_path,
    eog=False,
    emg=False,
    scaler="sample",
    session_name=1,
):
    """XXX docstring"""
    all_data = list()
    all_events = list()
    subjects_valid = list()
    rec_subjects = {}
    datatype = "eeg"
    suffix = "eeg"
    pbar = tqdm(total=n_subjects)
    all_sub = (
        pd.read_csv(
            data_path / "participants.tsv",
            delimiter="\t",
        )["participant_id"]
        .transform(lambda x: x[4:])
        .tolist()
    )

    for subj_id in range(len(all_sub)):
        subject = all_sub[subj_id]
        if len(subjects_valid) >= n_subjects:
            break
        try:
            if session_name == "phys":
                bids_path = BIDSPath(
                    datatype=datatype,
                    root=data_path,
                    suffix=suffix,
                    subject=subject,
                )
                ses = bids_path.match()[0].session
            else:
                ses = session_name

            bids_path = BIDSPath(
                datatype=datatype,
                root=data_path,
                suffix=suffix,
                task="sleep",
                session=ses,
                subject=subject,
            )
            data, events = read_raw_bids_with_preprocessing(
                bids_path, scaler, eog, emg
            )
            all_data.append(data)
            all_events.append(events)
            if subj_id not in subjects_valid:
                subjects_valid.append(subj_id)
                rec_subjects[subj_id] = 1

        except ValueError:
            print("That was no valid epoch.")

        except PermissionError:
            print("subject no valid")

        except TypeError:
            print("subject no valid")

        except FileNotFoundError:
            print("File not found")

        pbar.update(1)

    # Concatenate into a single dataset
    pbar.close()
    print(len(subjects_valid))
    return all_data, all_events, subjects_valid


def load_sleep_dataset(
    n_subjects,
    dataset_name,
    eog=False,
    emg=False,
    data_path=None,
    scaler="sample",
):
    data_path = DATA_PATH
    if dataset_name == "MASS":
        data_path = data_path / "MASS" / "SS3" / "7channels"
        ses = None
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "ABC":
        data_path = data_path / "ABC" / "7channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "CHAT":
        data_path = data_path / "CHAT" / "7channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "CFS":
        data_path = data_path / "CFS" / "2channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "HOMEPAP":
        data_path = data_path / "HOMEPAP" / "7channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "CCSHS":
        data_path = data_path / "CCSHS" / "2channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "SOF":
        data_path = data_path / "SOF" / "2channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "MROS":
        data_path = data_path / "MROS" / "2channels"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "Physionet":
        data_path = data_path / "Physionet" / "4channels-eeg_eog_emg"
        ses = "phys"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)

    if dataset_name == "SHHS":
        data_path = data_path / "SHHS" / "4channels-eeg_eog_emg"
        ses = "1"
        return load_data(n_subjects, data_path, eog, emg, scaler, ses)


def load_BCI_dataset(
    dataset_name,
    subject_id=None,
    n_jobs=1,
    cropped_decoding=False,
    input_windows_samples=1000,
    n_preds_per_input=467,
    exp_mov=False,
    filter=False,
    resample=None,
    channels_to_pick=None,
    mapping=None,
):
    if dataset_name in ["Shin2017A", "Shin2017B"]:
        dataset = MOABBDataset(
            dataset_name=dataset_name, subject_ids=subject_id, accept=True
        )
    else:
        dataset = MOABBDataset(
            dataset_name=dataset_name, subject_ids=subject_id
        )
    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 40.0  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    # Factor to convert from V to uV
    factor = 1e6

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, factor)),
    ]

    preprocessors += [
        Preprocessor(
            apply_on_array=False,
            fn="pick_channels",
            ch_names=channels_to_pick,
            ordered=True,
        ),
    ]
    if filter:
        preprocessors += [
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),
        ]
    if exp_mov:
        preprocessors += [
            Preprocessor(
                exponential_moving_standardize,
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]
    if resample:
        preprocessors += [
            Preprocessor(
                "resample", sfreq=resample
            ),
        ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]

    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    if cropped_decoding:
        window_size_samples = input_windows_samples
        window_stride_samples = n_preds_per_input
    else:
        window_size_samples = None
        window_stride_samples = None

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=False,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        mapping=mapping,
    )

    splitted = windows_dataset.split("subject")
    subjects = list(splitted.keys())
    X = []
    y = []
    for sub in subjects:
        n_domains = len(splitted[sub].datasets)
        X.append([
            splitted[sub].datasets[i].windows.get_data().astype(np.float32)
            for i in range(n_domains)
        ])
        y.append([splitted[sub].datasets[i].y for i in range(n_domains)])
    return X, y
