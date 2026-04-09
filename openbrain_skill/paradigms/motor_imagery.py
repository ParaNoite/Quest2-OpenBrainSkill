"""Motor Imagery paradigm handler."""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from openbrain_skill.paradigms.base import BaseParadigm


class MotorImageryParadigm(BaseParadigm):
    """Motor Imagery (MI) paradigm.

    Participants imagine movement of a limb. Discriminative patterns
    appear in mu (8–12 Hz) and beta (13–30 Hz) bands via ERD/ERS.

    Event codes:
        1 = Left hand imagery
        2 = Right hand imagery
        3 = Both feet imagery
        4 = Tongue imagery
    """

    paradigm_name = "motor_imagery"

    @property
    def needs_epoched_data(self) -> bool:
        """Motor imagery CSP requires epoched data."""
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
        """Build the MI preprocessing pipeline.

        Steps:
            1. Bandpass filter (0.5–40 Hz for EEG; 0.5–200 Hz for ECoG)
            2. Notch filter (50/60 Hz)
            3. Common average reference
            4. Epoching

        Args:
            modality: Signal modality (``"eeg"`` or ``"ecog"``).
            sfreq: Sampling frequency in Hz.
            **kwargs: Forwarded to pipeline steps (e.g., ``l_freq``, ``h_freq``).

        Returns:
            Configured :class:`~openbrain_skill.core.pipeline.Pipeline`.
        """
        from openbrain_skill.core.pipeline import Pipeline, PipelineStep
        from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter
        from openbrain_skill.preprocessing.artifacts import common_average_reference

        # Modality-specific defaults
        if modality == "ecog":
            l_freq = kwargs.get("l_freq", 0.5)
            h_freq = kwargs.get("h_freq", 200.0)
            notch_freqs = kwargs.get("notch_freqs", [50, 100, 150, 60, 120, 180])
        elif modality == "fnirs":
            l_freq = kwargs.get("l_freq", 0.01)
            h_freq = kwargs.get("h_freq", 0.1)
            notch_freqs = []
        else:  # eeg (default)
            l_freq = kwargs.get("l_freq", 0.5)
            h_freq = kwargs.get("h_freq", 40.0)
            notch_freqs = kwargs.get("notch_freqs", [50, 60])

        pipeline = Pipeline()

        # Bandpass
        pipeline.add_step(
            PipelineStep(
                name="bandpass_filter",
                func=bandpass_filter,
                params={"l_freq": l_freq, "h_freq": h_freq, "sfreq": sfreq},
                description=(
                    f"Apply a zero-phase bandpass filter between {l_freq} Hz and "
                    f"{h_freq} Hz to remove DC drift and high-frequency noise while "
                    "preserving the mu (8–12 Hz) and beta (13–30 Hz) bands relevant "
                    "for motor imagery."
                ),
                code_template=(
                    "from scipy.signal import butter, sosfiltfilt\n\n"
                    "def bandpass_filter(data, l_freq, h_freq, sfreq):\n"
                    "    nyq = sfreq / 2\n"
                    f"    sos = butter(5, [{l_freq / (sfreq/2):.4f}, {min(h_freq, sfreq/2 - 1) / (sfreq/2):.4f}],\n"
                    "                btype='bandpass', output='sos')\n"
                    "    return sosfiltfilt(sos, data, axis=-1)"
                ),
            )
        )

        # Notch (only if there are frequencies to notch)
        if notch_freqs:
            pipeline.add_step(
                PipelineStep(
                    name="notch_filter",
                    func=notch_filter,
                    params={"freqs": notch_freqs, "sfreq": sfreq},
                    description=(
                        f"Apply notch filters at {notch_freqs} Hz to suppress "
                        "power-line interference."
                    ),
                    code_template=(
                        "from scipy.signal import iirnotch, sosfiltfilt\n\n"
                        "def notch_filter(data, freqs, sfreq):\n"
                        "    for f in freqs:\n"
                        "        b, a = iirnotch(f, Q=30, fs=sfreq)\n"
                        "        data = sosfiltfilt(\n"
                        "            [[b[0], b[1], b[2], 1, a[1], a[2]]], data, axis=-1)\n"
                        "    return data"
                    ),
                )
            )

        # CAR (only for EEG / ECoG)
        if modality in ("eeg", "ecog"):
            pipeline.add_step(
                PipelineStep(
                    name="common_average_reference",
                    func=common_average_reference,
                    params={},
                    description=(
                        "Re-reference the signal to the common average of all channels "
                        "to reduce volume-conduction and common-mode noise."
                    ),
                    code_template=(
                        "def common_average_reference(data):\n"
                        "    # data: (n_channels, n_samples)\n"
                        "    return data - data.mean(axis=0, keepdims=True)"
                    ),
                )
            )

        return pipeline

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def get_feature_extractors(self, modality: str, sfreq: float) -> List[Any]:
        """Return feature extractors appropriate for motor imagery.

        For EEG/ECoG: band power (mu + beta) and CSP.
        For fNIRS: HRF amplitude and mean HbO.

        Args:
            modality: Signal modality.
            sfreq: Sampling frequency.

        Returns:
            List of feature extractor instances.
        """
        from openbrain_skill.features.spectral import BandPowerExtractor, PSDExtractor
        from openbrain_skill.features.spatial import CSPExtractor

        if modality == "fnirs":
            from openbrain_skill.features.temporal import MeanExtractor, SlopeExtractor
            return [MeanExtractor(), SlopeExtractor()]

        extractors = [
            BandPowerExtractor(
                sfreq=sfreq,
                bands={"mu": (8, 12), "beta": (13, 30)},
            ),
            PSDExtractor(sfreq=sfreq),
            CSPExtractor(n_components=6),
        ]

        if modality == "ecog":
            extractors.insert(
                0,
                BandPowerExtractor(
                    sfreq=sfreq,
                    bands={"high_gamma": (70, 150)},
                ),
            )

        return extractors

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def generate_events(
        self,
        n_samples: int,
        sfreq: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic MI trial events.

        Trials are spaced 6 s apart (1 s cue + 4 s imagery + 1 s rest).

        Args:
            n_samples: Total recording samples.
            sfreq: Sampling frequency.
            seed: Random seed.

        Returns:
            Array of shape ``(n_events, 3)``.
        """
        rng = np.random.default_rng(seed)
        trial_duration_s = 6.0
        trial_samples = int(trial_duration_s * sfreq)
        n_trials = int(n_samples // trial_samples)

        events = []
        for i in range(n_trials):
            onset = i * trial_samples + int(1.0 * sfreq)  # 1 s after trial start
            code = rng.integers(1, 5)  # codes 1-4
            events.append([onset, 0, int(code)])

        return np.array(events, dtype=int) if events else np.empty((0, 3), dtype=int)
