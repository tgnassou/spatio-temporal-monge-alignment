from torch.utils.data import Dataset
from torch.utils.data import Sampler

import random


class DomainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, domain, transform=None):
        self.X = X
        self.y = y
        self.domain = domain

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.domain[idx]


class DomainBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.domain_to_indices = {}
        for i, domain in enumerate(dataset.domain):
            domain = domain.item()
            if domain not in self.domain_to_indices.keys():
                self.domain_to_indices[domain] = []
            self.domain_to_indices[domain].append(i)
        self.domain_keys = list(self.domain_to_indices.keys())

    def __iter__(self):
        if self.shuffle:
            batches = []
            for domain in self.domain_keys:
                indices = self.domain_to_indices[domain]
                random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    new_batch = indices[i:i + self.batch_size]
                    if len(new_batch) == self.batch_size:
                        batches.append(new_batch)
            random.shuffle(batches)
            for batch in batches:
                for i in range(0, len(batch)):
                    yield batch[i]
        else:
            # TODO: implement
            for domain in self.domain_keys:
                indices = self.domain_to_indices[domain]
                for i in range(0, len(indices)):
                    yield indices[i]

    def __len__(self):
        total_samples = (
            sum(len(indices) for indices in self.domain_to_indices.values())
        )
        return total_samples // self.batch_size
