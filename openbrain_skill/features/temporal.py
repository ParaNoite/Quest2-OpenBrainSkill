"""Temporal feature extractors for brain signals."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from openbrain_skill.features.base import BaseExtractor


class MeanExtractor(BaseExtractor):
    """Compute the mean amplitude per channel.

    Useful for fNIRS haemodynamic response amplitude estimation.
    """

    name = "mean"

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute channel means.

        Args:
            data: Array ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Array of shape ``(n_channels, 1)`` or ``(n_epochs, n_channels, 1)``.
        """
        if data.ndim == 2:
            return data.mean(axis=-1, keepdims=True)
        elif data.ndim == 3:
            return data.mean(axis=-1, keepdims=True)
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    @property
    def feature_names(self):
        return ["mean"]


class SlopeExtractor(BaseExtractor):
    """Compute the linear slope of each channel's time series.

    The slope (fitted via ordinary least squares) indicates the trend
    direction and rate, useful for fNIRS HRF analysis.
    """

    name = "slope"

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute per-channel linear slope.

        Args:
            data: Array ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Array of shape ``(n_channels, 1)`` or ``(n_epochs, n_channels, 1)``.
        """
        if data.ndim == 2:
            return self._extract_2d(data)
        elif data.ndim == 3:
            return np.stack([self._extract_2d(epoch) for epoch in data])
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    def _extract_2d(self, data: np.ndarray) -> np.ndarray:
        n_channels, n_samples = data.shape
        x = np.arange(n_samples, dtype=float)
        x -= x.mean()

        # OLS slope = cov(x, y) / var(x)
        x_var = (x**2).mean()
        slopes = ((data * x) / x_var).mean(axis=-1, keepdims=True)
        return slopes

    @property
    def feature_names(self):
        return ["slope"]


class HjorthExtractor(BaseExtractor):
    """Compute Hjorth parameters: Activity, Mobility, and Complexity.

    Hjorth parameters are classical EEG temporal features derived from
    the signal and its first and second derivatives.

    - **Activity**: Variance of the signal (≈ power).
    - **Mobility**: Square root of variance ratio (1st derivative / signal).
    - **Complexity**: Ratio of Mobility of 2nd derivative to Mobility of 1st.
    """

    name = "hjorth"

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute Hjorth parameters per channel.

        Args:
            data: Array ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Array of shape ``(n_channels, 3)`` or
            ``(n_epochs, n_channels, 3)``.
            Columns: [activity, mobility, complexity].
        """
        if data.ndim == 2:
            return self._extract_2d(data)
        elif data.ndim == 3:
            return np.stack([self._extract_2d(epoch) for epoch in data])
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    def _extract_2d(self, data: np.ndarray) -> np.ndarray:
        n_channels = data.shape[0]
        d1 = np.diff(data, axis=-1)
        d2 = np.diff(d1, axis=-1)

        var0 = data.var(axis=-1)
        var1 = d1.var(axis=-1)
        var2 = d2.var(axis=-1)

        activity = var0
        mobility = np.where(var0 > 0, np.sqrt(var1 / np.clip(var0, 1e-12, None)), 0.0)
        mob1 = np.where(var1 > 0, np.sqrt(var2 / np.clip(var1, 1e-12, None)), 0.0)
        complexity = np.where(mobility > 0, mob1 / np.clip(mobility, 1e-12, None), 0.0)

        return np.stack([activity, mobility, complexity], axis=-1)

    @property
    def feature_names(self):
        return ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]


class ERPAmplitudeExtractor(BaseExtractor):
    """Compute mean ERP amplitude within a time window.

    Intended for ERP analysis (e.g. P300 amplitude).

    Args:
        window: Tuple ``(t_start, t_end)`` in seconds relative to epoch onset.
        sfreq: Sampling frequency (Hz).
        tmin: Epoch start time (seconds). Used to convert *window* to samples.
    """

    name = "erp_amplitude"

    def __init__(
        self,
        window: Tuple[float, float],
        sfreq: float,
        tmin: float = 0.0,
    ) -> None:
        self.window = window
        self.sfreq = sfreq
        self.tmin = tmin

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute mean amplitude in the ERP window.

        Args:
            data: Epoched array ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Array of shape ``(n_epochs, n_channels, 1)``.
        """
        if data.ndim != 3:
            raise ValueError(
                f"ERPAmplitudeExtractor expects 3-D epoched data; got {data.ndim}-D."
            )
        start_s = int(round((self.window[0] - self.tmin) * self.sfreq))
        end_s = int(round((self.window[1] - self.tmin) * self.sfreq))
        start_s = max(0, start_s)
        end_s = min(data.shape[-1], max(start_s + 1, end_s))

        return data[:, :, start_s:end_s].mean(axis=-1, keepdims=True)

    @property
    def feature_names(self):
        return ["erp_amplitude"]


class ERPLatencyExtractor(BaseExtractor):
    """Compute the latency of the positive peak within a time window.

    Intended for P300 latency analysis.

    Args:
        window: Tuple ``(t_start, t_end)`` in seconds relative to epoch onset.
        sfreq: Sampling frequency (Hz).
        tmin: Epoch start time (seconds).
    """

    name = "erp_latency"

    def __init__(
        self,
        window: Tuple[float, float],
        sfreq: float,
        tmin: float = 0.0,
    ) -> None:
        self.window = window
        self.sfreq = sfreq
        self.tmin = tmin

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute latency of the peak positive sample.

        Args:
            data: Epoched array ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Array of shape ``(n_epochs, n_channels, 1)`` with peak latency
            in seconds.
        """
        if data.ndim != 3:
            raise ValueError(
                f"ERPLatencyExtractor expects 3-D epoched data; got {data.ndim}-D."
            )
        start_s = int(round((self.window[0] - self.tmin) * self.sfreq))
        end_s = int(round((self.window[1] - self.tmin) * self.sfreq))
        start_s = max(0, start_s)
        end_s = min(data.shape[-1], max(start_s + 1, end_s))

        window_data = data[:, :, start_s:end_s]
        peak_idx = window_data.argmax(axis=-1)  # (n_epochs, n_channels)
        peak_latency_s = (peak_idx + start_s) / self.sfreq + self.tmin
        return peak_latency_s[..., np.newaxis]

    @property
    def feature_names(self):
        return ["erp_latency"]
