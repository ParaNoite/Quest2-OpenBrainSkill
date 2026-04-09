"""Resting-state paradigm handler."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from openbrain_skill.paradigms.base import BaseParadigm


class RestingStateParadigm(BaseParadigm):
    """Resting-state paradigm.

    No explicit task; spontaneous brain activity is recorded. Typical
    conditions are eyes-open (EO) and eyes-closed (EC).

    Event codes:
        1 = Eyes-open period start
        2 = Eyes-closed period start
    """

    paradigm_name = "resting_state"

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def get_preprocessing_pipeline(
        self,
        modality: str,
        sfreq: float,
        **kwargs: Any,
    ):
        """Build the resting-state preprocessing pipeline.

        Steps:
            1. Bandpass filter
            2. Notch filter (EEG/ECoG)
            3. Common average reference (EEG/ECoG)
            4. Segmentation into fixed-length windows

        Args:
            modality: Signal modality.
            sfreq: Sampling frequency.
            **kwargs: Optional parameter overrides.

        Returns:
            Configured Pipeline.
        """
        from openbrain_skill.core.pipeline import Pipeline, PipelineStep
        from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter
        from openbrain_skill.preprocessing.artifacts import common_average_reference
        from openbrain_skill.preprocessing.epoching import segment_into_windows

        if modality == "fnirs":
            l_freq = kwargs.get("l_freq", 0.01)
            h_freq = kwargs.get("h_freq", 0.1)
            notch_freqs: list = []
        elif modality == "ecog":
            l_freq = kwargs.get("l_freq", 0.5)
            h_freq = kwargs.get("h_freq", 200.0)
            notch_freqs = kwargs.get("notch_freqs", [50, 100, 150, 60, 120, 180])
        else:  # eeg
            l_freq = kwargs.get("l_freq", 1.0)
            h_freq = kwargs.get("h_freq", 45.0)
            notch_freqs = kwargs.get("notch_freqs", [50, 60])

        win_duration = kwargs.get("window_duration", 2.0)  # seconds
        win_overlap = kwargs.get("window_overlap", 0.5)

        pipeline = Pipeline()

        pipeline.add_step(
            PipelineStep(
                name="bandpass_filter",
                func=bandpass_filter,
                params={"l_freq": l_freq, "h_freq": h_freq, "sfreq": sfreq},
                description=(
                    f"Bandpass filter ({l_freq}–{h_freq} Hz) to preserve standard "
                    "frequency bands (delta, theta, alpha, beta, gamma) and remove "
                    "slow drifts and high-frequency noise."
                ),
                code_template=(
                    "from scipy.signal import butter, sosfiltfilt\n\n"
                    f"sos = butter(5, [{l_freq}/(sfreq/2), {h_freq}/(sfreq/2)],\n"
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
                    description=f"Notch filter at {notch_freqs} Hz.",
                    code_template=(
                        "from scipy.signal import iirnotch\n"
                        "for f in freqs:\n"
                        "    b, a = iirnotch(f, Q=30, fs=sfreq)\n"
                        "    data = filtfilt(b, a, data, axis=-1)"
                    ),
                )
            )

        if modality in ("eeg", "ecog"):
            pipeline.add_step(
                PipelineStep(
                    name="common_average_reference",
                    func=common_average_reference,
                    params={},
                    description="Re-reference to the common average of all channels.",
                    code_template=(
                        "def common_average_reference(data):\n"
                        "    return data - data.mean(axis=0, keepdims=True)"
                    ),
                )
            )

        pipeline.add_step(
            PipelineStep(
                name="segmentation",
                func=segment_into_windows,
                params={
                    "window_duration": win_duration,
                    "overlap": win_overlap,
                    "sfreq": sfreq,
                },
                description=(
                    f"Segment continuous signal into {win_duration} s windows "
                    f"with {win_overlap * 100:.0f} % overlap."
                ),
                code_template=(
                    "import numpy as np\n\n"
                    "def segment(data, duration, overlap, sfreq):\n"
                    "    win = int(duration * sfreq)\n"
                    "    step = int(win * (1 - overlap))\n"
                    "    starts = range(0, data.shape[-1] - win + 1, step)\n"
                    "    return np.stack([data[..., s:s+win] for s in starts])"
                ),
            )
        )

        return pipeline

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def get_feature_extractors(self, modality: str, sfreq: float) -> List[Any]:
        """Return feature extractors for resting state.

        Returns:
            Band power, spectral entropy, and Hjorth parameter extractors.
        """
        from openbrain_skill.features.spectral import BandPowerExtractor, SpectralEntropyExtractor
        from openbrain_skill.features.temporal import HjorthExtractor

        if modality == "fnirs":
            from openbrain_skill.features.spectral import PSDExtractor
            from openbrain_skill.features.temporal import MeanExtractor
            return [PSDExtractor(sfreq=sfreq), MeanExtractor()]

        if modality == "ecog":
            bands = {
                "delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
                "beta": (13, 30), "low_gamma": (30, 70), "high_gamma": (70, 150),
            }
        else:
            bands = {
                "delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
                "beta": (13, 30), "gamma": (30, 45),
            }

        return [
            BandPowerExtractor(sfreq=sfreq, bands=bands),
            SpectralEntropyExtractor(sfreq=sfreq),
            HjorthExtractor(),
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
        """Generate synthetic eyes-open / eyes-closed alternating events.

        Each block lasts 60 s; they alternate EO (code 1) and EC (code 2).

        Args:
            n_samples: Total recording samples.
            sfreq: Sampling frequency.
            seed: Random seed (unused; events are deterministic).

        Returns:
            Array of shape ``(n_events, 3)``.
        """
        block_samples = int(60.0 * sfreq)
        n_blocks = max(1, int(n_samples // block_samples))

        events = []
        for i in range(n_blocks):
            onset = i * block_samples
            code = (i % 2) + 1  # alternates 1, 2, 1, 2, ...
            events.append([onset, 0, code])

        return np.array(events, dtype=int)
