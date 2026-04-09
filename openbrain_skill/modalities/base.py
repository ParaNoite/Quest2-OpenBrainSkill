"""Abstract base class for signal modalities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseModality(ABC):
    """Abstract base for a brain signal modality.

    Subclasses must implement :meth:`generate_synthetic_data` and
    :meth:`validate_data`, and should set :attr:`modality_name`.

    Args:
        sfreq: Sampling frequency (Hz).
        n_channels: Number of recording channels.
    """

    modality_name: str = "base"

    def __init__(self, sfreq: float = 256.0, n_channels: int = 64) -> None:
        self.sfreq = sfreq
        self.n_channels = n_channels

    @abstractmethod
    def generate_synthetic_data(
        self,
        duration: float = 10.0,
        n_trials: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic data for testing and demonstrations.

        Args:
            duration: Recording duration in seconds.
            n_trials: Optional number of trials for epoched data.
            seed: Random seed for reproducibility.

        Returns:
            Synthetic signal array of shape ``(n_channels, n_samples)`` or
            ``(n_trials, n_channels, n_samples)`` when *n_trials* is given.
        """

    @abstractmethod
    def validate_data(self, data: np.ndarray) -> None:
        """Validate that *data* has the expected shape and dtype.

        Args:
            data: Signal array to validate.

        Raises:
            ValueError: If the data shape is incompatible.
        """

    @property
    def default_channel_names(self) -> List[str]:
        """Default channel name strings (can be overridden by subclasses)."""
        return [f"CH{i:03d}" for i in range(self.n_channels)]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sfreq={self.sfreq}, n_channels={self.n_channels})"
        )
