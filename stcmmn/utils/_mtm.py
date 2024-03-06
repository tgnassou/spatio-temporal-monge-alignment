import os
import numpy as np
from torch import nn


def load_a_dataset(data_path, domain_id, set="test"):
    fname_base = f"miniTimeMatch_{domain_id}{set}_interpolate.npy"
    fname_base_labels = f"miniTimeMatch_{domain_id}{set}_labels.npy"
    X = np.load(os.path.join(data_path, fname_base))
    y = np.load(os.path.join(
        data_path, fname_base_labels
    ))
    return X, y


class CNNmtm(nn.Module):
    def __init__(self, number_of_features, number_of_classes):
        super(CNNmtm, self).__init__()
        self.feature_extractor = nn.Sequential(
                nn.Conv1d(
                    in_channels=number_of_features, out_channels=128,
                    kernel_size=8, stride=1, bias=False, padding=3
                ),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU(),

                nn.Conv1d(
                    in_channels=128, out_channels=256, kernel_size=5,
                    stride=1, padding=2, bias=False
                ),
                nn.BatchNorm1d(num_features=256, affine=False),
                nn.ReLU(),

                nn.Conv1d(
                    in_channels=256, out_channels=128, kernel_size=3,
                    stride=1, padding=1, bias=False
                ),
                nn.BatchNorm1d(num_features=128, affine=False),
                nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(128, number_of_classes))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=-1)
        x = self.classifier(x)
        return x
