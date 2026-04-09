"""Abstract base class for experimental paradigms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseParadigm(ABC):
    """Abstract base for an experimental paradigm.

    Each paradigm defines how to build the preprocessing pipeline and
    feature extraction steps for a given modality.

    Subclasses must implement :meth:`get_preprocessing_pipeline` and
    :meth:`get_feature_extractors`.
    """

    paradigm_name: str = "base"

    @abstractmethod
    def get_preprocessing_pipeline(
        self,
        modality: str,
        sfreq: float,
        **kwargs: Any,
    ):
        """Build and return a :class:`~openbrain_skill.core.pipeline.Pipeline`
        for the given modality.

        Args:
            modality: Signal modality name.
            sfreq: Sampling frequency (Hz).
            **kwargs: Additional paradigm-specific parameters.

        Returns:
            A configured :class:`~openbrain_skill.core.pipeline.Pipeline`.
        """

    @abstractmethod
    def get_feature_extractors(self, modality: str, sfreq: float) -> List[Any]:
        """Return a list of feature extractor instances for this paradigm.

        Args:
            modality: Signal modality name.
            sfreq: Sampling frequency (Hz).

        Returns:
            List of feature extractor objects.
        """

    @abstractmethod
    def generate_events(
        self,
        n_samples: int,
        sfreq: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a synthetic event array compatible with this paradigm.

        Args:
            n_samples: Total number of samples in the recording.
            sfreq: Sampling frequency (Hz).
            seed: Random seed.

        Returns:
            Array of shape ``(n_events, 3)`` where columns are
            [sample_index, 0, event_code].
        """

    @property
    def needs_epoched_data(self) -> bool:
        """Whether feature extractors for this paradigm require epoched (3-D) data.

        Default is *False*; paradigms that use ERP or trial-based features
        should override this to *True*.
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
