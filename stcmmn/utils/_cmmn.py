import numpy as np

from scipy.signal import welch, csd
import scipy.fft as sp_fft
from abc import ABC
from joblib import Parallel, delayed

import ot


class CMMN(ABC):
    """Base class for CMMN."""

    def __init__(
        self,
        filter_size=128,
        method="temp",
        weights=None,
        eps=1e-4,
        reg=1e-7,
        num_iter=1,
        concatenate_epochs=True,
        n_jobs=1,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.weights = weights
        self.barycenter = None
        self.method = method
        self.eps = eps
        self.reg = reg
        self.num_iter = num_iter
        self.concatenate_epochs = concatenate_epochs
        self.n_jobs = n_jobs

    def fit(self, X):
        """Fit the barycenter.

        Parameters
        ----------
        X : list, shape=(K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.concatenate_epochs:
            # Reduce the number of dimension to (K, C, N*T)
            X = [np.concatenate(X[i], axis=-1) for i in range(len(X))]

        # Compute the power spectral density
        psd = self.compute_psd(X)

        # Compute the barycenter of the psd
        self.compute_barycenter(psd)

        return self

    def transform(self, X, return_H=False):
        """Transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        K = len(X)
        if self.concatenate_epochs:
            window_size = X[0].shape[-1]
            # Reduce the number of dimension to (K, C, N*T)
            X = [np.concatenate(X[i], axis=-1) for i in range(K)]

        # Compute the power spectral density
        psd = self.compute_psd(X)

        # Compute the filter and the convolution
        H = self.compute_filter(psd)
        X = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_convolution)(X[i], H[i]) for i in range(K)
        )

        if self.concatenate_epochs:
            # Reshape the data to (K, N, C, T)
            X = [self._epoching(X[i], window_size) for i in range(K)]
        if return_H:
            return X, H
        return X

    def fit_transform(self, X, return_H=False):
        """Fit the model and transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        K = len(X)
        if self.concatenate_epochs:
            window_size = X[0].shape[-1]
            # Reduce the number of dimension to (K, C, N*T)
            X = [np.concatenate(X[i], axis=-1) for i in range(K)]

        # Compute the power spectral density
        psd = self.compute_psd(X)

        # Compute the barycenter of the psd
        self.compute_barycenter(psd)

        # Compute the filter and the convolution
        H = self.compute_filter(psd)
        X = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_convolution)(X[i], H[i]) for i in range(K)
        )

        if self.concatenate_epochs:
            # Reshape the data to (K, N, C, T)
            X = [self._epoching(X[i], window_size) for i in range(K)]
        
        if return_H:
            return X, H
        return X

    def compute_barycenter(self, psd):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.method == "temp":
            self.barycenter = self._temporal_barycenter(psd)

        elif self.method == "spatiotemp":
            self.barycenter = self._spatio_temporal_barycenter(psd)

        elif self.method == "spatio":
            self.barycenter = ot.gaussian.bures_wasserstein_barycenter(
                np.zeros((len(psd), len(psd[0]))), psd
            )[1]

    def _temporal_barycenter(self, psd):
        K = len(psd)

        # Compute the weights for each domain
        if self.weights is None:
            weights = np.ones(psd.shape, dtype=psd[0].dtype) / K

        B = np.sum(weights * np.sqrt(psd), axis=0) ** 2
        return B

    def _fixed_point_barycenter(self, B, psd):
        K = len(psd)

        B_sqrt_ = self._matrix_operator(B.T, np.sqrt).T
        sum_B = np.einsum("ijl,njkl,kml -> niml", B_sqrt_, psd, B_sqrt_)
        sum_B_sqrt = np.array([
            self._matrix_operator(sum_B[i].T, np.sqrt).T for i in range(K)
        ])

        # Compute the weights for each domain
        if self.weights is None:
            weights = np.ones(sum_B_sqrt.shape, dtype=psd[0].dtype) / K

        B = np.sum(weights * np.sqrt(sum_B_sqrt), axis=0) ** 2
        return B

    def _spatio_temporal_barycenter(self, psd):
        diff = np.inf
        # init
        B = np.mean(psd, axis=0)

        # Fixed point iterations
        for _ in range(self.num_iter):
            B_new = self._fixed_point_barycenter(B, psd)
            diff = np.linalg.norm(B - B_new)
            B = B_new
            if diff <= self.eps:
                break
        else:
            print("Dit not converge.")
        return B

    def compute_filter(self, psd):
        """Compute filter to mapped the source data to the barycenter.

        This function compute the filter to mapped the source data to target
        frequency barycenter. One target need to be given to compute the
        filter.

        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.method == "temp":
            H = self._compute_temporal_filter(psd)
            H = np.fft.fftshift(H, axes=-1)

        elif self.method == "spatiotemp":
            H = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_spatio_temporal_filter)(psd[i])
                for i in range(len(psd))
            )
            H = np.fft.fftshift(H, axes=-1)

        if self.method == "spatio":
            m = np.zeros_like(psd[0][0])
            H = [
                ot.gaussian.bures_wasserstein_mapping(
                    m, m, psd[i], self.barycenter
                )[0]
                for i in range(len(psd))
            ]

        return H

    def _compute_temporal_filter(self, psd):
        if self.barycenter is None:
            raise ValueError("Barycenter need to be computed first")

        D = np.sqrt(self.barycenter) / np.sqrt(psd)
        H = sp_fft.irfftn(D, axes=-1)
        return H

    def _compute_spatio_temporal_filter(self, psd):
        psd_sqrt = self._matrix_operator(psd.T, np.sqrt).T
        psd_sqrt_inv = self._matrix_operator(psd.T, lambda x: 1 / np.sqrt(x)).T
        D_ = np.einsum(
            "ijl,jkl,kml -> iml", psd_sqrt, self.barycenter, psd_sqrt
        )
        D_sqrt_ = self._matrix_operator(D_.T, np.sqrt).T
        D = np.einsum(
            "ijl,jkl,kml -> iml", psd_sqrt_inv, D_sqrt_, psd_sqrt_inv
        )
        H = sp_fft.irfft(D, axis=-1)
        return H

    def compute_convolution(self, X, H):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(C, T) or (N, C, T)
            List of data signals for each domain.
        H : array, shape=(C, filter_size)
            Filters.
        """
        if self.method == "temp":
            if self.concatenate_epochs:
                X_norm = self._temporal_convolution(X, H)
            else:
                N = len(X)
                X_norm = [
                    self._temporal_convolution(X[i], H) for i in range(N)
                ]

        elif self.method == "spatiotemp":
            if self.concatenate_epochs:
                X_norm = self._spatio_temporal_convolution(X, H)
            else:
                N = len(X)
                X_norm = [
                    self._spatio_temporal_convolution(X[i], H)
                    for i in range(N)
                ]

        elif self.method == "spatio":
            if self.concatenate_epochs:
                X_norm = H @ X
            else:
                N = len(X)
                X_norm = [
                    H @ X[i]
                    for i in range(N)
                ]

        return X_norm

    def _temporal_convolution(self, X, H):
        C = len(X)
        X_norm = [
            np.convolve(X[chan], H[chan], mode="same") for chan in range(C)
        ]
        X_norm = np.array(X_norm)
        return X_norm

    def _spatio_temporal_convolution(self, X, H,):
        C = len(X)
        X_norm = np.array([
            np.sum(
                [np.convolve(X[j], H[i, j], mode="same") for j in range(C)],
                axis=0,
            )
            for i in range(len(H))
        ])
        return X_norm

    def compute_psd(self, X):
        """Compute the power spectral density of the data.

        Parameters
        ----------
        X : array, shape=(K, C, T) or (K, N, C, T)
            Data.

        Returns
        -------
        psd : array, shape=(K, C, filter_size) or (K, C, C, filter_size)
            Power spectral density or cross spectral density.
        """
        K = len(X)
        if self.method == "temp":
            if self.concatenate_epochs:
                psd = [
                    welch(X[i], nperseg=self.filter_size)[1]
                    for i in range(K)
                ]
            else:
                psd = []
                for i in range(K):
                    # Compute the average power spectral density over the
                    # samples of the domain
                    N = len(X[i])
                    psd.append(np.mean([
                        welch(X[i][j], nperseg=self.filter_size)[1]
                        for j in range(N)
                    ], axis=0))

        elif self.method == "spatiotemp":
            if self.concatenate_epochs:
                psd = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._get_csd)(X[i]) for i in range(K)
                )
            else:
                psd = []
                for i in range(K):
                    # Compute the average cross spectral density over the
                    # samples of the domain
                    N = len(X[i])
                    psd.append(np.mean(
                        [self._get_csd(X[i][j]) for j in range(N)],
                        axis=0
                    ))

        elif self.method == "spatio":
            if self.concatenate_epochs:
                psd = [
                    np.cov(X[i])
                    for i in range(K)
                ]
            else:
                psd = []
                for i in range(K):
                    # Compute the average covariance over the
                    # samples of the domain
                    N = len(X[i])
                    psd.append(np.mean(
                        [np.cov(X[i][j]) for j in range(N)],
                        axis=0
                    ))

        return np.array(psd)

    def _epoching(self, X, size):
        """Create a epoch of size `size` on the data `X`.

        Parameters
        ----------
        X : array, shape=(C, T)
            Data.
        size : int
            Size of the window.

        Returns
        -------
        array, shape=(n_epochs, C, size)
        """
        data = []
        start = 0
        end = size
        step = size
        length = X.shape[-1]
        while end <= length:
            data.append(X[:, start:end])
            start += step
            end += step
        return np.array(data)

    def _get_csd(self, X):
        psd = np.array([
            csd(X, X[i], nperseg=self.filter_size)[1]
            for i in range(len(X))
        ])
        return psd

    def _matrix_operator(self, A, operator, ensure_positive=True):
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.expand_dims(eigvals, -2)
        if ensure_positive:
            eigvals = eigvals.clip(0, None) + self.reg
        eigvals = operator(eigvals)
        A_operator = (eigvecs * eigvals) @ np.swapaxes(eigvecs.conj(), -2, -1)
        return A_operator
