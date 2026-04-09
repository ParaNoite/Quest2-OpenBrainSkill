"""Spectral feature extractors for brain signals."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from openbrain_skill.features.base import BaseExtractor


class BandPowerExtractor(BaseExtractor):
    """Compute mean power spectral density within defined frequency bands.

    Supports both continuous ``(n_channels, n_samples)`` and epoched
    ``(n_epochs, n_channels, n_samples)`` input.

    Args:
        sfreq: Sampling frequency (Hz).
        bands: Dict mapping band name → ``(low_hz, high_hz)`` tuple.
            Defaults to standard EEG bands.
        method: PSD estimation method. Currently only ``"welch"``.
        nperseg: Welch segment length in samples. Defaults to
            ``min(256, n_samples)``.
    """

    name = "band_power"

    _DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    def __init__(
        self,
        sfreq: float,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "welch",
        nperseg: Optional[int] = None,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands if bands is not None else dict(self._DEFAULT_BANDS)
        self.method = method
        self.nperseg = nperseg

    # ------------------------------------------------------------------

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute band power features.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            Feature array of shape ``(n_channels, n_bands)`` or
            ``(n_epochs, n_channels, n_bands)``.
        """
        if data.ndim == 2:
            return self._extract_2d(data)
        elif data.ndim == 3:
            return np.stack([self._extract_2d(epoch) for epoch in data])
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    def _extract_2d(self, data: np.ndarray) -> np.ndarray:
        """Compute band power for a single segment ``(n_channels, n_samples)``."""
        n_channels, n_samples = data.shape
        nperseg = self.nperseg or min(256, n_samples)

        freqs, psd = welch(data, fs=self.sfreq, nperseg=nperseg, axis=-1)

        band_names = list(self.bands)
        features = np.zeros((n_channels, len(band_names)))

        for j, band in enumerate(band_names):
            low, high = self.bands[band]
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                features[:, j] = psd[:, mask].mean(axis=-1)

        return features

    @property
    def feature_names(self) -> List[str]:
        """Return feature column names."""
        return [f"band_power_{b}" for b in self.bands]


class PSDExtractor(BaseExtractor):
    """Compute full power spectral density using Welch's method.

    Args:
        sfreq: Sampling frequency (Hz).
        nperseg: Welch segment length in samples.
        noverlap: Number of overlapping samples between segments.
    """

    name = "psd"

    def __init__(
        self,
        sfreq: float,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
    ) -> None:
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.noverlap = noverlap

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute PSD.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            PSD array of shape ``(n_channels, n_freqs)`` or
            ``(n_epochs, n_channels, n_freqs)``.
        """
        if data.ndim == 2:
            return self._extract_2d(data)
        elif data.ndim == 3:
            return np.stack([self._extract_2d(epoch) for epoch in data])
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    def _extract_2d(self, data: np.ndarray) -> np.ndarray:
        n_samples = data.shape[-1]
        nperseg = self.nperseg or min(256, n_samples)
        _, psd = welch(
            data,
            fs=self.sfreq,
            nperseg=nperseg,
            noverlap=self.noverlap,
            axis=-1,
        )
        return psd

    def get_frequencies(self, n_samples: int) -> np.ndarray:
        """Return the frequency axis for a given number of samples."""
        nperseg = self.nperseg or min(256, n_samples)
        freqs, _ = welch(np.zeros(n_samples), fs=self.sfreq, nperseg=nperseg)
        return freqs


class SpectralEntropyExtractor(BaseExtractor):
    """Compute spectral entropy (Shannon entropy of the normalised PSD).

    Spectral entropy measures the flatness of the power spectrum. Lower
    values indicate more rhythmic/peaked spectra; higher values indicate
    broadband/noise-like spectra.

    Args:
        sfreq: Sampling frequency (Hz).
        nperseg: Welch segment length.
    """

    name = "spectral_entropy"

    def __init__(self, sfreq: float, nperseg: Optional[int] = None) -> None:
        self.sfreq = sfreq
        self.nperseg = nperseg

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute spectral entropy per channel.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
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
        n_samples = data.shape[-1]
        nperseg = self.nperseg or min(256, n_samples)
        _, psd = welch(data, fs=self.sfreq, nperseg=nperseg, axis=-1)

        # Normalise PSD to a probability distribution
        psd_sum = psd.sum(axis=-1, keepdims=True)
        psd_norm = np.where(psd_sum > 0, psd / psd_sum, 0.0)

        # Shannon entropy: -sum(p * log2(p))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_psd = np.where(psd_norm > 0, np.log2(psd_norm), 0.0)
        entropy = -(psd_norm * log_psd).sum(axis=-1, keepdims=True)

        return entropy

    @property
    def feature_names(self) -> List[str]:
        return ["spectral_entropy"]


class SNRExtractor(BaseExtractor):
    """Compute signal-to-noise ratio at target frequencies (for SSVEP).

    For each target frequency, SNR is computed as the ratio of power at
    the target frequency to the mean power of neighbouring bins
    (excluding the target).

    Args:
        sfreq: Sampling frequency (Hz).
        target_frequencies: List of stimulus frequencies to assess (Hz).
        n_neighbour_bins: Number of neighbouring bins on each side for
            noise estimation.
        nperseg: Welch segment length.
    """

    name = "snr"

    def __init__(
        self,
        sfreq: float,
        target_frequencies: Optional[List[float]] = None,
        n_neighbour_bins: int = 4,
        nperseg: Optional[int] = None,
    ) -> None:
        self.sfreq = sfreq
        self.target_frequencies = target_frequencies or [7.5, 10.0, 12.0, 15.0]
        self.n_neighbour_bins = n_neighbour_bins
        self.nperseg = nperseg

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute SNR features.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Returns:
            SNR array of shape ``(n_channels, n_targets)`` or
            ``(n_epochs, n_channels, n_targets)``.
        """
        if data.ndim == 2:
            return self._extract_2d(data)
        elif data.ndim == 3:
            return np.stack([self._extract_2d(epoch) for epoch in data])
        else:
            raise ValueError(f"Expected 2-D or 3-D data; got {data.ndim}-D.")

    def _extract_2d(self, data: np.ndarray) -> np.ndarray:
        n_channels, n_samples = data.shape
        nperseg = self.nperseg or min(256, n_samples)
        freqs, psd = welch(data, fs=self.sfreq, nperseg=nperseg, axis=-1)

        snr = np.zeros((n_channels, len(self.target_frequencies)))

        for j, target_freq in enumerate(self.target_frequencies):
            # Find closest bin
            target_idx = np.argmin(np.abs(freqs - target_freq))
            # Noise bins: neighbours excluding target
            start = max(0, target_idx - self.n_neighbour_bins)
            end = min(len(freqs), target_idx + self.n_neighbour_bins + 1)
            noise_mask = np.arange(start, end) != target_idx

            noise_indices = np.arange(start, end)[noise_mask]
            if len(noise_indices) == 0:
                snr[:, j] = 0.0
            else:
                noise_power = psd[:, noise_indices].mean(axis=-1)
                signal_power = psd[:, target_idx]
                snr[:, j] = np.where(
                    noise_power > 0, signal_power / noise_power, 0.0
                )

        return snr

    @property
    def feature_names(self) -> List[str]:
        return [f"snr_{f:.1f}hz" for f in self.target_frequencies]
