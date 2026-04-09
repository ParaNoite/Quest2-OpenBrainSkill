"""Preprocessing package for OpenBrainSkill."""

from openbrain_skill.preprocessing.filters import bandpass_filter, notch_filter
from openbrain_skill.preprocessing.artifacts import (
    common_average_reference,
    threshold_artifact_rejection,
    apply_ica_simple,
)
from openbrain_skill.preprocessing.epoching import (
    epoch_data,
    baseline_correct,
    segment_into_windows,
)

__all__ = [
    "bandpass_filter",
    "notch_filter",
    "common_average_reference",
    "threshold_artifact_rejection",
    "apply_ica_simple",
    "epoch_data",
    "baseline_correct",
    "segment_into_windows",
]
