"""Tests for auto and teaching modes."""
import numpy as np
import pytest

from openbrain_skill.core.skill import BrainSkill
from openbrain_skill.modes.auto_mode import AutoMode
from openbrain_skill.modes.teaching_mode import TeachingMode


def make_skill(modality="eeg", paradigm="resting_state", sfreq=64.0, n_channels=4):
    return BrainSkill(
        modality=modality,
        paradigm=paradigm,
        sfreq=sfreq,
        n_channels=n_channels,
    )


# ---------------------------------------------------------------------------
# AutoMode
# ---------------------------------------------------------------------------

class TestAutoMode:
    def test_run_with_synthetic_data(self):
        skill = make_skill()
        mode = AutoMode(skill=skill, verbose=False)
        result = mode.run()
        assert "feature_matrix" in result
        assert "report" in result
        assert "metadata" in result
        assert isinstance(result["feature_matrix"], np.ndarray)

    def test_run_with_explicit_data(self):
        skill = make_skill()
        data = skill.modality_handler.generate_synthetic_data(duration=5.0, seed=99)
        mode = AutoMode(skill=skill, verbose=False)
        result = mode.run(data)
        assert result["feature_matrix"].size > 0

    def test_metadata_fields(self):
        skill = make_skill()
        mode = AutoMode(skill=skill, verbose=False)
        result = mode.run()
        meta = result["metadata"]
        assert meta["modality"] == "eeg"
        assert meta["paradigm"] == "resting_state"
        assert meta["sfreq"] == 64.0
        assert meta["n_channels"] == 4

    def test_report_has_required_keys(self):
        skill = make_skill()
        mode = AutoMode(skill=skill, verbose=False)
        result = mode.run()
        report = result["report"]
        assert "timestamp" in report
        assert "pipeline" in report
        assert "features" in report
        assert "metadata" in report

    def test_feature_names_list(self):
        skill = make_skill()
        mode = AutoMode(skill=skill, verbose=False)
        result = mode.run()
        assert isinstance(result["feature_names"], list)
        assert len(result["feature_names"]) == result["feature_matrix"].shape[-1]

    def test_all_modality_paradigm_combos(self):
        combos = [
            ("eeg", "motor_imagery", 64.0, 4),
            ("eeg", "p300", 64.0, 4),
            ("eeg", "ssvep", 64.0, 4),
            ("eeg", "resting_state", 64.0, 4),
            ("fnirs", "resting_state", 10.0, 6),
            ("fnirs", "motor_imagery", 10.0, 6),
            ("ecog", "motor_imagery", 256.0, 8),
            ("ecog", "resting_state", 256.0, 8),
        ]
        for modality, paradigm, sfreq, n_ch in combos:
            skill = make_skill(modality, paradigm, sfreq, n_ch)
            mode = AutoMode(skill=skill, verbose=False)
            result = mode.run()
            assert result["feature_matrix"].ndim >= 1, (
                f"Failed for {modality}/{paradigm}"
            )


# ---------------------------------------------------------------------------
# TeachingMode
# ---------------------------------------------------------------------------

class TestTeachingMode:
    def test_run_returns_expected_keys(self, capsys):
        skill = make_skill()
        mode = TeachingMode(skill=skill)
        result = mode.run()
        for key in ("steps", "explanations", "code_templates", "knowledge",
                    "references", "feature_matrix", "feature_names", "report"):
            assert key in result, f"Missing key: {key}"

    def test_steps_have_structure(self, capsys):
        skill = make_skill()
        mode = TeachingMode(skill=skill)
        result = mode.run()
        steps = result["steps"]
        assert isinstance(steps, list)
        assert len(steps) > 0
        for step in steps:
            assert "name" in step
            assert "description" in step
            assert "code_template" in step

    def test_explanations_is_string(self, capsys):
        skill = make_skill()
        mode = TeachingMode(skill=skill)
        result = mode.run()
        assert isinstance(result["explanations"], str)

    def test_code_templates_is_string(self, capsys):
        skill = make_skill()
        mode = TeachingMode(skill=skill)
        result = mode.run()
        assert isinstance(result["code_templates"], str)
        assert "openbrain_skill" in result["code_templates"]

    def test_feature_matrix_not_empty(self, capsys):
        skill = make_skill()
        mode = TeachingMode(skill=skill)
        result = mode.run()
        assert result["feature_matrix"].size > 0

    def test_eeg_motor_imagery_teaching(self, capsys):
        skill = make_skill("eeg", "motor_imagery")
        mode = TeachingMode(skill=skill)
        result = mode.run()
        # Motor imagery should mention CSP or band power in steps
        step_names = [s["name"] for s in result["steps"]]
        assert "bandpass_filter" in step_names

    def test_with_explicit_data(self, capsys):
        skill = make_skill()
        data = skill.modality_handler.generate_synthetic_data(duration=5.0, seed=7)
        mode = TeachingMode(skill=skill)
        result = mode.run(data=data)
        assert result["feature_matrix"].size > 0
