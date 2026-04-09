"""Tests for the core pipeline and skill modules."""
import numpy as np
import pytest

from openbrain_skill.core.pipeline import Pipeline, PipelineStep
from openbrain_skill.core.skill import BrainSkill
from openbrain_skill.core.knowledge_graph import KnowledgeGraph
from openbrain_skill.core.registry import Registry, get_default_registry


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineStep:
    def test_run_applies_function(self):
        def double(data):
            return data * 2

        step = PipelineStep(name="double", func=double)
        data = np.ones((4, 100))
        result = step.run(data)
        np.testing.assert_array_equal(result, data * 2)

    def test_run_passes_params(self):
        def add_offset(data, offset=0):
            return data + offset

        step = PipelineStep(name="add", func=add_offset, params={"offset": 5})
        data = np.zeros((4, 100))
        result = step.run(data)
        np.testing.assert_array_equal(result, np.full_like(data, 5))

    def test_repr(self):
        step = PipelineStep(name="my_step", func=lambda x: x)
        assert "my_step" in repr(step)


class TestPipeline:
    def _make_step(self, name, func=None):
        if func is None:
            func = lambda x: x
        return PipelineStep(name=name, func=func)

    def test_add_and_run_steps(self):
        pipeline = Pipeline()
        pipeline.add_step(self._make_step("a", lambda x: x + 1))
        pipeline.add_step(self._make_step("b", lambda x: x * 2))

        data = np.zeros((2, 50))
        result, info = pipeline.run(data)
        # (0 + 1) * 2 = 2
        np.testing.assert_array_equal(result, np.full_like(data, 2))
        assert "a" in info
        assert "b" in info

    def test_remove_step(self):
        pipeline = Pipeline()
        pipeline.add_step(self._make_step("x"))
        pipeline.add_step(self._make_step("y"))
        pipeline.remove_step("x")
        assert len(pipeline) == 1
        assert pipeline.steps[0].name == "y"

    def test_remove_nonexistent_raises(self):
        pipeline = Pipeline()
        with pytest.raises(KeyError):
            pipeline.remove_step("ghost")

    def test_get_step(self):
        pipeline = Pipeline()
        step = self._make_step("find_me")
        pipeline.add_step(step)
        assert pipeline.get_step("find_me") is step

    def test_step_timings_recorded(self):
        pipeline = Pipeline()
        pipeline.add_step(self._make_step("slow", lambda x: x))
        data = np.zeros((2, 50))
        _, _ = pipeline.run(data)
        assert "slow" in pipeline.step_timings
        assert pipeline.step_timings["slow"] >= 0

    def test_chaining(self):
        p = Pipeline()
        result = p.add_step(self._make_step("a")).add_step(self._make_step("b"))
        assert result is p
        assert len(p) == 2

    def test_repr(self):
        p = Pipeline()
        p.add_step(self._make_step("alpha"))
        assert "alpha" in repr(p)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_get_modality(self):
        registry = Registry()
        registry.register_modality("test_mod", dict)  # use dict as dummy class
        assert registry.get_modality("test_mod") is dict

    def test_register_duplicate_modality_raises(self):
        registry = Registry()
        registry.register_modality("dup", dict)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_modality("dup", list)

    def test_get_unknown_modality_raises(self):
        registry = Registry()
        with pytest.raises(KeyError, match="Unknown modality"):
            registry.get_modality("ghost")

    def test_register_and_get_paradigm(self):
        registry = Registry()
        registry.register_paradigm("test_par", dict)
        assert registry.get_paradigm("test_par") is dict

    def test_list_modalities(self):
        registry = Registry()
        registry.register_modality("a", dict)
        registry.register_modality("b", list)
        assert set(registry.list_modalities()) == {"a", "b"}

    def test_default_registry_has_built_ins(self):
        registry = get_default_registry()
        assert "eeg" in registry.list_modalities()
        assert "fnirs" in registry.list_modalities()
        assert "ecog" in registry.list_modalities()
        assert "motor_imagery" in registry.list_paradigms()
        assert "p300" in registry.list_paradigms()
        assert "ssvep" in registry.list_paradigms()
        assert "resting_state" in registry.list_paradigms()


# ---------------------------------------------------------------------------
# KnowledgeGraph tests
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    def test_query_known_combination(self):
        kg = KnowledgeGraph()
        entry = kg.query("eeg", "motor_imagery")
        assert entry is not None
        assert "preprocessing" in entry
        assert "features" in entry

    def test_query_unknown_returns_none(self):
        kg = KnowledgeGraph()
        assert kg.query("eeg", "ghost_paradigm") is None

    def test_get_preprocessing_steps(self):
        kg = KnowledgeGraph()
        steps = kg.get_preprocessing_steps("eeg", "motor_imagery")
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_get_features(self):
        kg = KnowledgeGraph()
        feats = kg.get_features("eeg", "motor_imagery")
        assert isinstance(feats, list)
        assert any(f["name"] == "band_power" for f in feats)

    def test_get_description(self):
        kg = KnowledgeGraph()
        desc = kg.get_description("eeg", "p300")
        assert "P300" in desc or "ERP" in desc

    def test_list_combinations(self):
        kg = KnowledgeGraph()
        combos = kg.list_combinations()
        assert "eeg:motor_imagery" in combos
        assert "fnirs:resting_state" in combos
        assert "ecog:motor_imagery" in combos

    def test_add_custom_entry(self):
        kg = KnowledgeGraph()
        kg.add_entry("eeg", "custom_paradigm", {"description": "test", "preprocessing": [], "features": []})
        assert kg.query("eeg", "custom_paradigm") is not None

    def test_json_roundtrip(self):
        kg = KnowledgeGraph()
        json_str = kg.to_json()
        kg2 = KnowledgeGraph.from_json(json_str)
        assert kg2.list_combinations() == kg.list_combinations()

    def test_to_dict(self):
        kg = KnowledgeGraph()
        d = kg.to_dict()
        assert isinstance(d, dict)
        assert "eeg:motor_imagery" in d


# ---------------------------------------------------------------------------
# BrainSkill tests
# ---------------------------------------------------------------------------

class TestBrainSkill:
    def test_create_eeg_motor_imagery(self):
        skill = BrainSkill(modality="eeg", paradigm="motor_imagery", n_channels=8)
        assert skill.modality == "eeg"
        assert skill.paradigm == "motor_imagery"

    def test_create_fnirs_resting_state(self):
        skill = BrainSkill(
            modality="fnirs", paradigm="resting_state", sfreq=10.0, n_channels=10
        )
        assert skill.modality == "fnirs"
        assert skill.sfreq == 10.0

    def test_unknown_modality_raises(self):
        with pytest.raises(KeyError):
            BrainSkill(modality="unknown_mod", paradigm="motor_imagery")

    def test_unknown_paradigm_raises(self):
        with pytest.raises(KeyError):
            BrainSkill(modality="eeg", paradigm="unknown_par")

    def test_describe_returns_string(self):
        skill = BrainSkill(modality="eeg", paradigm="p300", n_channels=8)
        desc = skill.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_knowledge_returns_dict(self):
        skill = BrainSkill(modality="eeg", paradigm="ssvep", n_channels=8)
        kg = skill.get_knowledge()
        assert isinstance(kg, dict)

    def test_get_preprocessing_steps(self):
        skill = BrainSkill(modality="eeg", paradigm="motor_imagery", n_channels=8)
        steps = skill.get_preprocessing_steps()
        assert len(steps) > 0

    def test_repr(self):
        skill = BrainSkill(modality="eeg", paradigm="p300", n_channels=4)
        r = repr(skill)
        assert "eeg" in r
        assert "p300" in r

    def test_run_auto_returns_feature_matrix(self):
        skill = BrainSkill(
            modality="eeg",
            paradigm="resting_state",
            sfreq=64.0,
            n_channels=4,
        )
        result = skill.run_auto(verbose=False)
        assert "feature_matrix" in result
        assert isinstance(result["feature_matrix"], np.ndarray)
        assert "report" in result
        assert "metadata" in result

    def test_run_auto_with_explicit_data(self):
        skill = BrainSkill(
            modality="eeg",
            paradigm="motor_imagery",
            sfreq=64.0,
            n_channels=4,
        )
        data = skill.modality_handler.generate_synthetic_data(duration=10.0, seed=0)
        result = skill.run_auto(data=data, verbose=False)
        assert result["feature_matrix"].size > 0
