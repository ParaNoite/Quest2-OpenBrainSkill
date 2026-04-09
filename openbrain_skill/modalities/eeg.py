"""EEG modality handler."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from openbrain_skill.modalities.base import BaseModality


# Standard 10-20 channel layout (subset for common cap sizes)
_EEG_CHANNELS_64 = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7",  "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",  "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7",  "P5",  "P3",  "P1",  "Pz",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1",  "Oz",  "O2",
    "Iz",
]


class EEGModality(BaseModality):
    """Handler for Electroencephalography (EEG) signals.

    Typical sampling frequencies range from 250 Hz to 2048 Hz.
    Signals are measured in microvolts (µV).

    Args:
        sfreq: Sampling frequency (Hz). Default: 256 Hz.
        n_channels: Number of EEG channels. Default: 64.
    """

    modality_name = "eeg"

    def __init__(self, sfreq: float = 256.0, n_channels: int = 64) -> None:
        super().__init__(sfreq=sfreq, n_channels=n_channels)

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        duration: float = 10.0,
        n_trials: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Generate realistic synthetic EEG data.

        The signal is composed of:
        - 1/f (pink noise) background activity
        - Alpha (10 Hz) oscillations on occipital channels
        - Beta (20 Hz) oscillations on central channels

        Args:
            duration: Duration in seconds.
            n_trials: If given, return epoched array ``(n_trials, n_ch, n_samp)``.
            seed: Random seed.

        Returns:
            Array of shape ``(n_channels, n_samples)`` or
            ``(n_trials, n_channels, n_samples)``.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(duration * self.sfreq)
        t = np.arange(n_samples) / self.sfreq

        # White noise base (µV scale)
        data = rng.normal(0, 5.0, (self.n_channels, n_samples))

        # Add pink noise (1/f) via cumulative sum normalisation
        pink = np.cumsum(rng.normal(0, 1.0, (self.n_channels, n_samples)), axis=1)
        pink -= pink.mean(axis=1, keepdims=True)
        pink *= 3.0 / (pink.std(axis=1, keepdims=True) + 1e-12)
        data += pink

        # Alpha on last quarter channels (occipital-like)
        occ_start = max(0, self.n_channels - self.n_channels // 4)
        alpha = 8.0 * np.sin(2 * np.pi * 10.0 * t)
        data[occ_start:] += alpha

        # Beta on middle channels (central-like)
        cen_start = self.n_channels // 4
        cen_end = self.n_channels // 2
        beta = 4.0 * np.sin(2 * np.pi * 20.0 * t)
        data[cen_start:cen_end] += beta

        if n_trials is not None:
            trial_len = n_samples
            trials = np.stack(
                [
                    self.generate_synthetic_data(
                        duration=duration, n_trials=None, seed=seed + i if seed is not None else None
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
        """Validate EEG data shape.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Raises:
            ValueError: If shape is incompatible.
        """
        if data.ndim == 2:
            if data.shape[0] != self.n_channels:
                raise ValueError(
                    f"EEG data must have {self.n_channels} channels, "
                    f"got {data.shape[0]}."
                )
        elif data.ndim == 3:
            if data.shape[1] != self.n_channels:
                raise ValueError(
                    f"Epoched EEG data must have {self.n_channels} channels "
                    f"(axis 1), got {data.shape[1]}."
                )
        else:
            raise ValueError(
                f"EEG data must be 2-D or 3-D, got {data.ndim}-D."
            )

    # ------------------------------------------------------------------
    # Channel names
    # ------------------------------------------------------------------

    @property
    def default_channel_names(self) -> List[str]:
        """Return standard 10-20 channel names (truncated/extended as needed)."""
        names = list(_EEG_CHANNELS_64)
        if self.n_channels <= len(names):
            return names[: self.n_channels]
        # Extend beyond 64 channels with generic names
        extras = [f"CH{i:03d}" for i in range(len(names), self.n_channels)]
        return names + extras
