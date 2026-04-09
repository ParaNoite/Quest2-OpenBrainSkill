"""ECoG (Electrocorticography) modality handler."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from openbrain_skill.modalities.base import BaseModality


class ECoGModality(BaseModality):
    """Handler for Electrocorticography (ECoG) signals.

    ECoG electrodes are placed directly on the cortical surface, providing
    high spatial resolution and broad bandwidth (0.5–500 Hz). Signals are
    measured in microvolts (µV).

    Args:
        sfreq: Sampling frequency (Hz). Default: 1000 Hz.
        n_channels: Number of ECoG electrodes. Default: 128.
    """

    modality_name = "ecog"

    def __init__(self, sfreq: float = 1000.0, n_channels: int = 128) -> None:
        super().__init__(sfreq=sfreq, n_channels=n_channels)

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        duration: float = 5.0,
        n_trials: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Generate synthetic ECoG data.

        Simulates:
        - Broadband 1/f background activity
        - High-gamma (80 Hz) bursts on a subset of channels
        - Beta (20 Hz) oscillations
        - Spike-like transients

        Args:
            duration: Duration in seconds.
            n_trials: If given, return epoched array.
            seed: Random seed.

        Returns:
            Array of shape ``(n_channels, n_samples)`` or
            ``(n_trials, n_channels, n_samples)``.
            Values in µV.
        """
        rng = np.random.default_rng(seed)
        n_samples = int(duration * self.sfreq)
        t = np.arange(n_samples) / self.sfreq

        # 1/f background
        white = rng.normal(0, 30.0, (self.n_channels, n_samples))
        # Approximate 1/f by cumulative sum (simplified)
        pink = np.cumsum(rng.normal(0, 1.0, (self.n_channels, n_samples)), axis=1)
        pink -= pink.mean(axis=1, keepdims=True)
        pink *= 10.0 / (pink.std(axis=1, keepdims=True) + 1e-12)
        data = white + pink

        # High-gamma bursts on first quarter channels
        hg_end = max(1, self.n_channels // 4)
        hg = 15.0 * np.sin(2 * np.pi * 80.0 * t)
        # Amplitude modulation to create burst-like pattern
        envelope = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None)
        data[:hg_end] += hg * envelope

        # Beta oscillations on middle channels
        mid_start = self.n_channels // 4
        mid_end = self.n_channels // 2
        beta = 8.0 * np.sin(2 * np.pi * 20.0 * t)
        data[mid_start:mid_end] += beta

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
        """Validate ECoG data shape.

        Args:
            data: Array of shape ``(n_channels, n_samples)`` or
                ``(n_epochs, n_channels, n_samples)``.

        Raises:
            ValueError: If shape is incompatible.
        """
        if data.ndim == 2:
            if data.shape[0] != self.n_channels:
                raise ValueError(
                    f"ECoG data must have {self.n_channels} channels, "
                    f"got {data.shape[0]}."
                )
        elif data.ndim == 3:
            if data.shape[1] != self.n_channels:
                raise ValueError(
                    f"Epoched ECoG data must have {self.n_channels} channels "
                    f"(axis 1), got {data.shape[1]}."
                )
        else:
            raise ValueError(f"ECoG data must be 2-D or 3-D, got {data.ndim}-D.")

    # ------------------------------------------------------------------
    # Channel names
    # ------------------------------------------------------------------

    @property
    def default_channel_names(self) -> List[str]:
        """Return ECoG electrode grid names."""
        n_cols = 16
        names = []
        for i in range(self.n_channels):
            row = i // n_cols + 1
            col = i % n_cols + 1
            names.append(f"G{row:02d}{col:02d}")
        return names
