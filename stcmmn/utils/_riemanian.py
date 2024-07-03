from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.covariance import covariances
from sklearn.covariance import empirical_covariance, shrunk_covariance
import numpy as np
import scipy

from abc import ABC


class RiemanianAlignment(ABC):
    def __init__(self, mean_update=True, non_homogeneous=False):
        super().__init__()
        self.mean_update = mean_update
        self.non_homogeneous = non_homogeneous

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

        if len(X[0].shape) == 3:
            X_new = [
                np.array([
                    np.matmul(cov_mean, X[i][j])
                    for j in range(len(X[i]))
                ])
                for i in range(len(X))
            ]
        else:
            X_new = [
                np.matmul(cov_mean, X[i])
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
        if len(X[0].shape) == 3:

            X_new = [
                np.array([
                    np.matmul(self.cov_mean, X[i][j])
                    for j in range(len(X[i]))
                ])
                for i in range(len(X))
            ]
        else:
            X_new = [
                np.matmul(self.cov_mean, X[i])
                for i in range(len(X))
            ]
        return X_new

    def _compute_mean_covariance(self, X):
        if len(X[0].shape) == 3:
            X = np.concatenate(X, axis=0)
        if self.non_homogeneous:
            cov = np.array([
                covariances(X[i][np.newaxis, :, :], estimator="oas")[0]
                for i in range(len(X))
            ])
        else:
            cov = np.array([
                covariances(X[i][np.newaxis, :, :], estimator="oas")[0]
                for i in range(len(X))
            ])

        eye = np.eye(cov.shape[1])*1e-7
        cov = cov + eye
        cov_mean = mean_covariance(cov, metric="riemann")
        cov_mean = np.linalg.inv(scipy.linalg.sqrtm(cov_mean))
        return cov_mean
