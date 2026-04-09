"""SSVEP paradigm handler."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from openbrain_skill.paradigms.base import BaseParadigm


class SSVEPParadigm(BaseParadigm):
    """Steady-State Visual Evoked Potential (SSVEP) paradigm.

    A flickering visual stimulus drives a resonance response at the
    stimulation frequency and harmonics, primarily over occipital channels.

    Args:
        stimulus_frequencies: List of stimulus flicker frequencies (Hz).
            Default: ``[7.5, 10.0, 12.0, 15.0]``.
    """

    paradigm_name = "ssvep"

    def __init__(self, stimulus_frequencies: Optional[List[float]] = None) -> None:
        self.stimulus_frequencies = stimulus_frequencies or [7.5, 10.0, 12.0, 15.0]

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def get_preprocessing_pipeline(
        self,
        modality: str,
        sfreq: float,
        **kwargs: Any,
    ):
        """Build the SSVEP preprocessing pipeline.

        Steps:
            1. Bandpass filter (1–50 Hz)
            2. Notch filter (50/60 Hz, if not overlapping stimulus frequencies)
            3. Segmentation / epoching

        Args:
            modality: Signal modality.
            sfreq: Sampling frequency.
            **kwargs: Optional parameter overrides.

        Returns:
            Configured Pipeline.
        """
        from openbrain_skill.core.pipeline import Pipeline, PipelineStep
        from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter

        l_freq = kwargs.get("l_freq", 1.0)
        h_freq = kwargs.get("h_freq", 50.0)
        notch_freqs = kwargs.get("notch_freqs", [50, 60])

        # Exclude notch frequencies that overlap with stimulus or harmonics
        safe_notch = [
            f
            for f in notch_freqs
            if all(abs(f - sf) > 2 for sf in self.stimulus_frequencies)
        ]

        pipeline = Pipeline()

        pipeline.add_step(
            PipelineStep(
                name="bandpass_filter",
                func=bandpass_filter,
                params={"l_freq": l_freq, "h_freq": h_freq, "sfreq": sfreq},
                description=(
                    f"Bandpass filter ({l_freq}–{h_freq} Hz) to preserve SSVEP "
                    f"frequency components at {self.stimulus_frequencies} Hz and "
                    "their second harmonics."
                ),
                code_template=(
                    "from scipy.signal import butter, sosfiltfilt\n\n"
                    "sos = butter(5, [1.0/(sfreq/2), 50.0/(sfreq/2)],\n"
                    "             btype='bandpass', output='sos')\n"
                    "filtered = sosfiltfilt(sos, data, axis=-1)"
                ),
            )
        )

        if safe_notch:
            pipeline.add_step(
                PipelineStep(
                    name="notch_filter",
                    func=notch_filter,
                    params={"freqs": safe_notch, "sfreq": sfreq},
                    description=(
                        f"Notch filter at {safe_notch} Hz. "
                        "Frequencies too close to stimulus frequencies are skipped."
                    ),
                    code_template=(
                        "from scipy.signal import iirnotch\n"
                        "b, a = iirnotch(50, Q=30, fs=sfreq)\n"
                        "data = filtfilt(b, a, data, axis=-1)"
                    ),
                )
            )

        return pipeline

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def get_feature_extractors(self, modality: str, sfreq: float) -> List[Any]:
        """Return feature extractors for SSVEP.

        Returns:
            PSD and SNR extractors.
        """
        from openbrain_skill.features.spectral import PSDExtractor, SNRExtractor

        return [
            PSDExtractor(sfreq=sfreq),
            SNRExtractor(sfreq=sfreq, target_frequencies=self.stimulus_frequencies),
        ]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def generate_events(
        self,
        n_samples: int,
        sfreq: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic SSVEP paradigm events.

        Each trial lasts 5 s; stimulus is presented at one of the configured
        frequencies in a round-robin fashion.

        Args:
            n_samples: Total recording samples.
            sfreq: Sampling frequency.
            seed: Random seed.

        Returns:
            Array of shape ``(n_events, 3)``.
        """
        trial_samples = int(5.0 * sfreq)
        n_trials = int(n_samples // trial_samples)

        events = []
        for i in range(n_trials):
            onset = i * trial_samples
            # Cycle through stimulus frequencies
            freq_idx = i % len(self.stimulus_frequencies)
            # Encode frequency index as event code (1-based)
            code = freq_idx + 1
            events.append([onset, 0, code])

        return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)
