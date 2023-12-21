import numpy as np

import scipy.signal
import scipy.fft as sp_fft
from abc import ABC
from joblib import Parallel, delayed


class CMMN(ABC):
    """Base class for CMMN."""
    def __init__(
        self,
        filter_size=128,
        fs=100,
        method="temp",
        weights=None,
        eps=1e-4,
        reg=1e-4,
        num_iter=1,
        concatenate_epochs=True,
        n_jobs=1,
    ):
        super().__init__()
        self.filter_size = filter_size
        self.fs = fs
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
        self.compute_barycenter(X)

        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.concatenate_epochs:
            window_size = X[0].shape[-1]
            # Reduce the number of dimension to (K, C, N*T)
            X = [np.concatenate(X[i], axis=-1) for i in range(len(X))]
            H = self.compute_filter(X)
            X = self.compute_convolution(X, H)
            X = [self._epoching(X[i], window_size) for i in range(len(X))]
        else:
            H = self.compute_filter(X)
            X = [self.compute_convolution(X[i], H[i]) for i in range(len(X))]

        return X

    def fit_transform(self, X):
        """Fit the model and transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        self.fit(X)
        return self.transform(X)

    def compute_barycenter(self, X):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.method == "temp":
            self.barycenter = self._temporal_barycenter(X)
        elif self.method == "spatiotemp":
            self.barycenter = self._spatio_temporal_barycenter(X)

    def _temporal_barycenter(self, X):
        K = len(X)
        psd = [
            scipy.signal.welch(X[i], nperseg=self.filter_size, fs=self.fs)[1]
            for i in range(K)
        ]
        psd = np.array(psd)
        if self.weights is None:
            weights = np.ones(psd.shape, dtype=X[0].dtype) / K

        barycenter = np.sum(weights * np.sqrt(psd), axis=0) ** 2
        return barycenter

    def _fixed_point_barycenter(self, Bbar, B):
        K = len(B)
        Bbar_sqrt_ = self._matrix_operator(Bbar.T, np.sqrt).T

        sum_B = np.einsum("ijl,njkl,kml -> niml", Bbar_sqrt_, B, Bbar_sqrt_)
        sum_B_sqrt = [
            self._matrix_operator(sum_B[i].T, np.sqrt).T for i in range(K)
        ]

        return np.mean(sum_B_sqrt, axis=0)

    def _spatio_temporal_barycenter(self, X):
        K = len(X)
        if self.concatenate_epochs:
            B = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_csd)(X[i]) for i in range(K)
            )
        else:
            B = []
            for i in range(K):
                B += [self._get_csd(X[i][j]) for j in range(len(X[i]))]
        diff = np.inf
        Bbar = np.mean(B, axis=0)
        for _ in range(self.num_iter):
            Bbar_new = self._fixed_point_barycenter(Bbar, B)
            diff = np.linalg.norm(Bbar - Bbar_new)
            Bbar = Bbar_new
            if diff <= self.eps:
                break
        else:
            print("Dit not converge.")

        return Bbar

    def compute_filter(self, X):
        """Compute filter to mapped the source data to the barycenter.

        This function compute the filter to mapped the source data to target
        frequency barycenter. One target need to be given to compute the
        filter.

        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if self.method == "temp":
            H = self._compute_temporal_filter(X)
        elif self.method == "spatiotemp":
            H = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_spatio_temporal_filter)(X[i])
                for i in range(len(X))
            )
        return np.fft.fftshift(H, axes=-1)

    def _compute_temporal_filter(self, X):
        K = len(X)
        psd = [
            scipy.signal.welch(X[i], nperseg=self.filter_size, fs=self.fs)[1]
            for i in range(K)
        ]

        if self.barycenter is None:
            raise ValueError("Barycenter need to be computed first")
        D = np.sqrt(self.barycenter) / np.sqrt(psd)
        H = sp_fft.irfftn(D, axes=-1)

        return H

    def _compute_spatio_temporal_filter(self, X):
        if self.concatenate_epochs:
            B = self._get_csd(X)
        else:
            B = np.mean([self._get_csd(X[i]) for i in range(len(X))], axis=0)
        B_sqrt = self._matrix_operator(B.T, np.sqrt).T
        B_sqrt_inv = self._matrix_operator(B.T, lambda x: 1 / np.sqrt(x)).T
        D_ = np.einsum("ijl,jkl,kml -> iml", B_sqrt, self.barycenter, B_sqrt)
        D_sqrt_ = self._matrix_operator(D_.T, np.sqrt).T
        D = np.einsum("ijl,jkl,kml -> iml", B_sqrt_inv, D_sqrt_, B_sqrt_inv)

        H = sp_fft.irfft(D, axis=-1)
        return H

    def compute_convolution(self, X, H):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        H : array, shape=(K, C, filter_size)
            Filters.
        """
        if self.method == "temp":
            X_norm = [
                self._temporal_convolution(X[i], H[i]) for i in range(len(X))
            ]
        elif self.method == "spatiotemp":
            if self.concatenate_epochs:
                X_norm = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._spatio_temporal_convolution)(X[i], H[i])
                    for i in range(len(X))
                )
            else:
                X_norm = [
                    self._spatio_temporal_convolution(X[i], H)
                    for i in range(len(X))
                ]
        return X_norm

    def _temporal_convolution(self, X, H):
        X_norm = [
            np.convolve(X[chan], H[chan], mode="same")
            for chan in range(len(H))
        ]
        X_norm = np.array(X_norm)
        return X_norm

    def _spatio_temporal_convolution(self, X, H,):
        X_norm = np.array([
            np.sum(
                [
                    np.convolve(X[j], H[i, j], mode="same")
                    for j in range(len(X))
                ],
                axis=0,
            )
            for i in range(len(H))
        ])
        return X_norm

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

    def _get_csd(self, X, ensure_positive=True):
        csd = np.array([
            scipy.signal.csd(X, X[i], nperseg=self.filter_size, fs=self.fs)[1]
            for i in range(len(X))
        ])
        return csd

    def _matrix_operator(self, A, operator, ensure_positive=True):
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.expand_dims(eigvals, -2)
        if ensure_positive:
            eigvals = eigvals.clip(0, None) + self.reg
        eigvals = operator(eigvals)
        A_operator = (eigvecs * eigvals) @ np.swapaxes(eigvecs.conj(), -2, -1)
        return A_operator
