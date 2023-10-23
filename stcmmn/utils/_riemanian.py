from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.covariance import covariances

import numpy as np
import scipy

from abc import ABC


class RiemanianAlignment(ABC):
    def __init__(self, mean_update=True):
        super().__init__()
        self.mean_update = mean_update

    def fit(self, X):
        """Compute the mean covariance.

        Parameters
        ----------
        X : list, shape=(K, N, C, T) or (N x T, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        self.cov_mean = self._compute_mean_covariance(X)
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : list, shape=(K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.mean_update:
            cov_mean = self._compute_mean_covariance(X)
        else:
            cov_mean = self.cov_mean
        X_new = [
            np.array([
                np.matmul(cov_mean, X[i][j])
                for j in range(len(X[i]))
            ])
            for i in range(len(X))
        ]
        return X_new

    def fit_transform(self, X):
        """Fit the mean covariance and transform the data.

        Parameters
        ----------
        X : list, shape=(K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        self.cov_mean = self._compute_mean_covariance(X)

        X_new = [
            np.array([
                np.matmul(self.cov_mean, X[i][j])
                for j in range(len(X[i]))
            ])
            for i in range(len(X))
        ]
        return X_new

    def _compute_mean_covariance(self, X):
        if len(X[0].shape) == 3:
            X = np.concatenate(X, axis=0)
        cov = covariances(X, estimator="oas")
        cov_mean = mean_covariance(cov, metric="riemann")
        cov_mean = np.linalg.inv(scipy.linalg.sqrtm(cov_mean))
        return cov_mean
