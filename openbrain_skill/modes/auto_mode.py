"""Auto mode: fully automated pipeline execution for OpenBrainSkill."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


class AutoMode:
    """Automatic mode that runs the full preprocessing + feature extraction pipeline.

    Given raw brain signal data and a configured :class:`~openbrain_skill.core.skill.BrainSkill`,
    this mode:

    1. Validates the input data.
    2. Builds the modality- and paradigm-specific preprocessing pipeline.
    3. Runs all preprocessing steps.
    4. Applies feature extractors.
    5. Returns a feature matrix and a structured report.

    Args:
        skill: A configured :class:`~openbrain_skill.core.skill.BrainSkill` instance.
        verbose: If *True*, print progress to stdout.
    """

    def __init__(self, skill: Any, verbose: bool = True) -> None:
        self._skill = skill
        self.verbose = verbose

    # ------------------------------------------------------------------

    def run(
        self,
        data: Optional[np.ndarray] = None,
        *,
        channel_names: Optional[List[str]] = None,
        events: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline.

        Args:
            data: Raw signal array ``(n_channels, n_samples)``. If *None*,
                synthetic data is generated automatically.
            channel_names: Optional channel name strings.
            events: Optional event array ``(n_events, 3)``.

        Returns:
            Dict with keys:
            - ``"feature_matrix"``: ``np.ndarray`` of extracted features.
            - ``"feature_names"``: List of feature column names.
            - ``"report"``: Structured report dict.
            - ``"metadata"``: Processing metadata dict.
        """
        t_total_start = time.perf_counter()
        skill = self._skill

        # 1. Generate synthetic data if none provided
        if data is None:
            if self.verbose:
                print(
                    f"[AutoMode] No data provided — generating synthetic "
                    f"{skill.modality.upper()} data..."
                )
            data = skill.modality_handler.generate_synthetic_data(seed=42)

        # 2. Validate data
        skill.modality_handler.validate_data(data)
        n_channels, n_samples = data.shape[:2] if data.ndim >= 2 else (1, data.shape[0])

        if channel_names is None:
            channel_names = skill.modality_handler.default_channel_names

        # 3. Generate events if none provided
        if events is None:
            if self.verbose:
                print("[AutoMode] No events provided — generating synthetic events...")
            sfreq = skill.sfreq
            n_samp = data.shape[-1]
            events = skill.paradigm_handler.generate_events(
                n_samples=n_samp, sfreq=sfreq
            )

        # 4. Build preprocessing pipeline
        if self.verbose:
            print(
                f"[AutoMode] Building preprocessing pipeline for "
                f"{skill.modality}/{skill.paradigm}..."
            )
        pipeline = skill.paradigm_handler.get_preprocessing_pipeline(
            modality=skill.modality, sfreq=skill.sfreq
        )

        # 5. Run preprocessing pipeline
        if self.verbose:
            print(f"[AutoMode] Running {len(pipeline)} preprocessing steps...")
        preprocessed_data, step_info = pipeline.run(data, verbose=self.verbose)

        # 5b. Epoch data if the paradigm's feature extractors require it
        if skill.paradigm_handler.needs_epoched_data and events is not None and preprocessed_data.ndim == 2:
            preprocessed_data = self._epoch_for_features(
                preprocessed_data, events, skill.sfreq
            )

        # 6. Extract features
        if self.verbose:
            print("[AutoMode] Extracting features...")
        extractors = skill.paradigm_handler.get_feature_extractors(
            modality=skill.modality, sfreq=skill.sfreq
        )

        feature_matrix, feature_names = self._run_extractors(
            preprocessed_data, extractors
        )

        t_total = time.perf_counter() - t_total_start

        # 7. Build report
        from openbrain_skill.report.generator import ReportGenerator

        generator = ReportGenerator()
        report = generator.generate(
            modality=skill.modality,
            paradigm=skill.paradigm,
            n_channels=n_channels,
            n_samples=n_samples,
            sfreq=skill.sfreq,
            channel_names=channel_names,
            n_events=len(events),
            pipeline_steps=[s.name for s in pipeline.steps],
            step_timings=pipeline.step_timings,
            feature_names=feature_names,
            feature_matrix=feature_matrix,
            knowledge_entry=skill.get_knowledge(),
            total_time_s=t_total,
        )

        if self.verbose:
            print(
                f"[AutoMode] Done in {t_total:.2f} s. "
                f"Feature matrix shape: {feature_matrix.shape}"
            )

        return {
            "feature_matrix": feature_matrix,
            "feature_names": feature_names,
            "report": report,
            "metadata": {
                "modality": skill.modality,
                "paradigm": skill.paradigm,
                "sfreq": skill.sfreq,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "n_events": len(events),
                "pipeline_steps": [s.name for s in pipeline.steps],
                "step_timings": pipeline.step_timings,
                "total_time_s": t_total,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _epoch_for_features(
        self,
        data: np.ndarray,
        events: np.ndarray,
        sfreq: float,
        tmin: float = 0.0,
        tmax: float = 2.0,
    ) -> np.ndarray:
        """Epoch continuous data around event onsets for feature extraction.

        Falls back to the full data reshaped as a single epoch if epoching
        produces no valid epochs.

        Args:
            data: Preprocessed continuous data ``(n_channels, n_samples)``.
            events: Event array ``(n_events, 3)``.
            sfreq: Sampling frequency.
            tmin: Epoch start time (s).
            tmax: Epoch end time (s).

        Returns:
            Epoched array ``(n_epochs, n_channels, epoch_samples)`` or
            ``(1, n_channels, n_samples)`` as fallback.
        """
        from openbrain_skill.preprocessing.epoching import epoch_data

        epochs, _ = epoch_data(data, events, tmin=tmin, tmax=tmax, sfreq=sfreq)
        if epochs.shape[0] == 0:
            # Fallback: wrap entire signal as a single epoch
            return data[np.newaxis, :, :]
        return epochs

    def _run_extractors(
        self, data: np.ndarray, extractors: list
    ):
        """Apply each extractor and concatenate features.

        Args:
            data: Preprocessed signal array.
            extractors: List of feature extractor instances.

        Returns:
            Tuple of ``(feature_matrix, feature_names)``.
        """
        all_features: List[np.ndarray] = []
        all_names: List[str] = []

        for extractor in extractors:
            try:
                feats = extractor.extract(data)
                # Flatten to 2-D: (n_channels_or_epochs, n_feats) or (n_feats,)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                elif feats.ndim == 3:
                    # (n_epochs, n_channels, n_feats) → (n_epochs, n_ch * n_feats)
                    n_epochs = feats.shape[0]
                    feats = feats.reshape(n_epochs, -1)

                all_features.append(feats)

                # Build feature names
                n_cols = feats.shape[-1]
                try:
                    names = extractor.feature_names
                    if len(names) < n_cols:
                        names = names + [
                            f"{extractor.name}_{i}" for i in range(len(names), n_cols)
                        ]
                    all_names.extend(names[:n_cols])
                except AttributeError:
                    all_names.extend(
                        [f"{extractor.name}_{i}" for i in range(n_cols)]
                    )

            except Exception as exc:
                if self.verbose:
                    print(
                        f"[AutoMode] Warning: extractor {extractor.name!r} "
                        f"failed: {exc}"
                    )

        if not all_features:
            return np.empty((0, 0)), []

        # Align row counts (take min)
        min_rows = min(f.shape[0] for f in all_features)
        all_features = [f[:min_rows] for f in all_features]

        feature_matrix = np.concatenate(all_features, axis=-1)
        return feature_matrix, all_names
