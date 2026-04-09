"""P300 paradigm handler."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from openbrain_skill.paradigms.base import BaseParadigm


class P300Paradigm(BaseParadigm):
    """P300 Event-Related Potential paradigm.

    A rare target stimulus elicits a positive ERP deflection (~300 ms
    post-stimulus) over parieto-central electrodes. Used in speller BCIs
    and cognitive studies.

    Event codes:
        1 = Target (rare, ~20 %)
        2 = Non-target (frequent, ~80 %)
    """

    paradigm_name = "p300"

    @property
    def needs_epoched_data(self) -> bool:
        """P300 feature extractors require epoched data."""
        return True

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def get_preprocessing_pipeline(
        self,
        modality: str,
        sfreq: float,
        **kwargs: Any,
    ):
        """Build the P300 preprocessing pipeline.

        Steps:
            1. Bandpass filter (0.1–30 Hz)
            2. Notch filter (50/60 Hz)
            3. Epoching

        Args:
            modality: Signal modality.
            sfreq: Sampling frequency.
            **kwargs: Optional overrides (``l_freq``, ``h_freq``).

        Returns:
            Configured Pipeline.
        """
        from openbrain_skill.core.pipeline import Pipeline, PipelineStep
        from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter

        l_freq = kwargs.get("l_freq", 0.1)
        h_freq = kwargs.get("h_freq", 30.0)
        notch_freqs = kwargs.get("notch_freqs", [50, 60])

        pipeline = Pipeline()

        pipeline.add_step(
            PipelineStep(
                name="bandpass_filter",
                func=bandpass_filter,
                params={"l_freq": l_freq, "h_freq": h_freq, "sfreq": sfreq},
                description=(
                    f"Apply a bandpass filter ({l_freq}–{h_freq} Hz) to preserve ERP "
                    "morphology. The P300 is a slow positive deflection; a low lower-cutoff "
                    "frequency is critical to avoid distorting the waveform."
                ),
                code_template=(
                    "from scipy.signal import butter, sosfiltfilt\n\n"
                    "sos = butter(5, [0.1 / (sfreq/2), 30.0 / (sfreq/2)],\n"
                    "             btype='bandpass', output='sos')\n"
                    "filtered = sosfiltfilt(sos, data, axis=-1)"
                ),
            )
        )

        if notch_freqs:
            pipeline.add_step(
                PipelineStep(
                    name="notch_filter",
                    func=notch_filter,
                    params={"freqs": notch_freqs, "sfreq": sfreq},
                    description="Remove power-line interference at 50/60 Hz.",
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
        """Return feature extractors for P300.

        Returns:
            List containing ERP amplitude and latency extractors.
        """
        from openbrain_skill.features.temporal import ERPAmplitudeExtractor, ERPLatencyExtractor

        return [
            ERPAmplitudeExtractor(window=(0.25, 0.45), sfreq=sfreq),
            ERPLatencyExtractor(window=(0.25, 0.55), sfreq=sfreq),
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
        """Generate synthetic P300 paradigm events.

        Stimuli are presented every 0.625 s (ISI). Approximately 20 % of
        stimuli are targets (code 1) and 80 % are non-targets (code 2).

        Args:
            n_samples: Total recording samples.
            sfreq: Sampling frequency.
            seed: Random seed.

        Returns:
            Array of shape ``(n_events, 3)``.
        """
        rng = np.random.default_rng(seed)
        isi_samples = int(0.625 * sfreq)
        n_stimuli = int(n_samples // isi_samples)

        events = []
        for i in range(n_stimuli):
            onset = i * isi_samples
            code = 1 if rng.random() < 0.2 else 2
            events.append([onset, 0, code])

        return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)
