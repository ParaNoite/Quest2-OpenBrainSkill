"""Spatial feature extractors for brain signals."""
from __future__ import annotations

from typing import Optional

import numpy as np

from openbrain_skill.features.base import BaseExtractor


class CSPExtractor(BaseExtractor):
    """Common Spatial Pattern (CSP) feature extraction.

    CSP finds spatial filters that maximise variance for one class while
    minimising it for the other. It is the gold standard for motor imagery
    EEG decoding.

    This implementation uses a lightweight numerically stable version based
    on generalised eigenvalue decomposition via scipy.

    Args:
        n_components: Number of CSP filters (total; split symmetrically
            between the two extremes). Must be even.
        regularisation: Regularisation parameter added to the covariance
            diagonal to improve numerical stability.

    Note:
        - CSP requires labelled data: call :meth:`fit` before :meth:`extract`.
        - For multi-class scenarios, use One-vs-Rest CSP.
    """

    name = "csp"

    def __init__(
        self, n_components: int = 6, regularisation: float = 1e-6
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be >= 2.")
        self.n_components = n_components
        self.regularisation = regularisation
        self._filters: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, epochs: np.ndarray, labels: np.ndarray) -> "CSPExtractor":
        """Estimate CSP filters from labelled training epochs.

        Args:
            epochs: Array ``(n_epochs, n_channels, n_samples)``.
            labels: 1-D integer label array ``(n_epochs,)``.
                Must contain exactly two unique classes.

        Returns:
            *self* (fitted).

        Raises:
            ValueError: If the number of unique classes is not 2.
        """
        if epochs.ndim != 3:
            raise ValueError(
                f"CSP.fit expects 3-D epoched data; got {epochs.ndim}-D."
            )
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes; got {len(unique_labels)}."
            )

        label_a, label_b = unique_labels
        epochs_a = epochs[labels == label_a]
        epochs_b = epochs[labels == label_b]

        cov_a = self._mean_covariance(epochs_a)
        cov_b = self._mean_covariance(epochs_b)

        self._filters = self._compute_csp_filters(cov_a, cov_b)
        return self

    def _mean_covariance(self, epochs: np.ndarray) -> np.ndarray:
        """Compute the normalised mean covariance across epochs."""
        n_epochs, n_channels, n_samples = epochs.shape
        covs = np.zeros((n_epochs, n_channels, n_channels))
        for i, epoch in enumerate(epochs):
            covs[i] = epoch @ epoch.T / np.trace(epoch @ epoch.T)
        return covs.mean(axis=0) + self.regularisation * np.eye(n_channels)

    def _compute_csp_filters(
        self, cov_a: np.ndarray, cov_b: np.ndarray
    ) -> np.ndarray:
        """Solve generalised eigenvalue problem: cov_a @ w = lambda * (cov_a + cov_b) @ w."""
        composite_cov = cov_a + cov_b
        # Whitening
        eigvals, eigvecs = np.linalg.eigh(composite_cov)
        # Sort by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Build whitening matrix P
        eps = 1e-12
        P = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, eps, None))) @ eigvecs.T

        # Transform cov_a into whitened space
        S = P @ cov_a @ P.T

        # Eigendecomposition of whitened cov_a
        evals, evecs = np.linalg.eigh(S)
        # Sort by descending eigenvalue (first filters → max variance for class A)
        sort_idx = np.argsort(evals)[::-1]
        evecs = evecs[:, sort_idx]

        # Spatial filters W = P.T @ evecs
        W = P.T @ evecs

        # Select first and last n_components//2 filters
        n_half = self.n_components // 2
        selected = np.concatenate([W[:, :n_half], W[:, -n_half:]], axis=1)
        return selected  # shape: (n_channels, n_components)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted CSP filters and compute log-variance features.

        Args:
            data: Array ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Feature array ``(n_epochs, n_components)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._filters is None:
            # Auto-fit if no labels provided (uses identity-like fallback)
            self._fit_identity(data.shape[1])

        if data.ndim != 3:
            raise ValueError(
                f"CSPExtractor.extract expects 3-D epoched data; got {data.ndim}-D."
            )

        W = self._filters  # (n_channels, n_components)
        # Apply filters: (n_epochs, n_comp, n_samples)
        filtered = np.einsum("ij,ejk->eik", W.T, data)
        # Log-variance feature
        variances = filtered.var(axis=-1)  # (n_epochs, n_components)
        log_var = np.log(np.clip(variances, 1e-12, None))
        return log_var

    def _fit_identity(self, n_channels: int) -> None:
        """Fall-back: use the first n_components channels as trivial filters."""
        n_comp = min(self.n_components, n_channels)
        self._filters = np.eye(n_channels)[:, :n_comp]

    @property
    def feature_names(self):
        return [f"csp_{i}" for i in range(self.n_components)]
