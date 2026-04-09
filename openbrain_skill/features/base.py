"""Abstract base class for feature extractors."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseExtractor(ABC):
    """Abstract base for feature extractors.

    Subclasses must implement :meth:`extract` and set :attr:`name`.
    """

    name: str = "base_extractor"

    @abstractmethod
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract features from *data*.

        Args:
            data: Signal array of shape ``(n_channels, n_samples)``,
                ``(n_epochs, n_channels, n_samples)``, or as required by
                the specific extractor.

        Returns:
            Feature array. Shape depends on the extractor:
            - For continuous data → ``(n_channels, n_features)``
            - For epoched data → ``(n_epochs, n_features)``
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
