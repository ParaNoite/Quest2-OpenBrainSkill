"""fNIRS modality handler."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from openbrain_skill.modalities.base import BaseModality


class FNIRSModality(BaseModality):
    """Handler for functional Near-Infrared Spectroscopy (fNIRS) signals.

    fNIRS measures haemodynamic responses via changes in HbO (oxyhaemoglobin)
    and HbR (deoxyhaemoglobin) concentrations. Typical sampling frequencies
    are 10–50 Hz.

    Args:
        sfreq: Sampling frequency (Hz). Default: 10 Hz.
        n_channels: Number of fNIRS source-detector channels (per wavelength).
            Default: 20 channels.
    """

    modality_name = "fnirs"

    def __init__(self, sfreq: float = 10.0, n_channels: int = 20) -> None:
        super().__init__(sfreq=sfreq, n_channels=n_channels)

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        duration: float = 60.0,
        n_trials: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Generate synthetic fNIRS HbO/HbR data.

        The simulated signal includes:
        - Slow haemodynamic fluctuations (0.01–0.1 Hz)
        - Cardiac artifact (~1 Hz)
        - Respiration artifact (~0.25 Hz)
        - Gaussian noise

        Args:
            duration: Duration in seconds.
            n_trials: If given, return epoched array.
            seed: Random seed.

        Returns:
            Array of shape ``(n_channels, n_samples)`` or
            ``(n_trials, n_channels, n_samples)``.
            Values represent HbO concentrations in µmol/L.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(duration * self.sfreq)
        t = np.arange(n_samples) / self.sfreq

        data = np.zeros((self.n_channels, n_samples))

        for ch in range(self.n_channels):
            # Haemodynamic response (slow, ~0.05 Hz)
            hemo = 0.5 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
            # Cardiac pulsation
            cardiac = 0.1 * np.sin(2 * np.pi * 1.1 * t + rng.uniform(0, 2 * np.pi))
            # Respiration
            resp = 0.15 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
            # Noise
            noise = rng.normal(0, 0.05, n_samples)
            data[ch] = hemo + cardiac + resp + noise

        if n_trials is not None:
            trials = np.stack(
                [
                    self.generate_synthetic_data(
                        duration=duration, n_trials=None, seed=(seed + i) if seed is not None else None
                    )
                    for i in range(n_trials)
                ]
            )
            return trials

        return data.astype(np.float64)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_data(self, data: np.ndarray) -> None:
        """Validate fNIRS data shape.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Raises:
            ValueError: If shape is incompatible.
        """
        if data.ndim == 2:
            if data.shape[0] != self.n_channels:
                raise ValueError(
                    f"fNIRS data must have {self.n_channels} channels, "
                    f"got {data.shape[0]}."
                )
        elif data.ndim == 3:
            if data.shape[1] != self.n_channels:
                raise ValueError(
                    f"Epoched fNIRS data must have {self.n_channels} channels "
                    f"(axis 1), got {data.shape[1]}."
                )
        else:
            raise ValueError(f"fNIRS data must be 2-D or 3-D, got {data.ndim}-D.")

    # ------------------------------------------------------------------
    # Channel names
    # ------------------------------------------------------------------

    @property
    def default_channel_names(self) -> List[str]:
        """Return source-detector pair channel names (HbO)."""
        return [f"S{(i // 2) + 1}_D{(i % 2) + 1} HbO" for i in range(self.n_channels)]
