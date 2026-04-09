"""Teaching mode: interactive step-by-step explanations with code templates."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class TeachingMode:
    """Teaching mode for the OpenBrainSkill framework.

    Provides:
    - Step-by-step explanations of the preprocessing pipeline.
    - Reusable, copy-pasteable Python code templates for each step.
    - Knowledge graph descriptions and literature references.
    - Optional execution of each step so the learner can inspect
      intermediate results.

    Args:
        skill: A configured :class:`~openbrain_skill.core.skill.BrainSkill`.
    """

    def __init__(self, skill: Any) -> None:
        self._skill = skill

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        data: Optional[np.ndarray] = None,
        *,
        channel_names: Optional[List[str]] = None,
        events: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline in teaching mode.

        Args:
            data: Optional raw signal array. If *None*, synthetic data is
                generated for demonstration.
            channel_names: Optional channel name list.
            events: Optional event array.

        Returns:
            Dict with keys:
            - ``"steps"``: List of step dicts (name, description, code_template).
            - ``"explanations"``: Human-readable narrative of the full pipeline.
            - ``"code_templates"``: Consolidated code template for the pipeline.
            - ``"knowledge"``: Knowledge graph entry for this modality/paradigm.
            - ``"references"``: Literature references.
            - ``"feature_matrix"``: Extracted feature matrix.
            - ``"feature_names"``: Feature column names.
            - ``"report"``: Structured processing report.
        """
        skill = self._skill

        # ---- Synthetic data if needed ----
        if data is None:
            print(
                f"[TeachingMode] No data provided — generating synthetic "
                f"{skill.modality.upper()} data for demonstration..."
            )
            data = skill.modality_handler.generate_synthetic_data(seed=42)

        skill.modality_handler.validate_data(data)
        n_channels = data.shape[0] if data.ndim >= 2 else 1
        n_samples = data.shape[-1]

        if channel_names is None:
            channel_names = skill.modality_handler.default_channel_names

        # ---- Events ----
        if events is None:
            events = skill.paradigm_handler.generate_events(
                n_samples=n_samples, sfreq=skill.sfreq
            )

        # ---- Build pipeline ----
        pipeline = skill.paradigm_handler.get_preprocessing_pipeline(
            modality=skill.modality, sfreq=skill.sfreq
        )

        # ---- Collect step details ----
        steps_info = self._collect_step_info(pipeline)

        # ---- Print teaching narrative ----
        self._print_narrative(pipeline, skill)

        # ---- Execute pipeline ----
        preprocessed_data, _step_info = pipeline.run(data)
        step_timing = pipeline.step_timings

        # ---- Epoch if required ----
        from openbrain_skill.modes.auto_mode import AutoMode
        _auto = AutoMode(skill=skill, verbose=False)

        if skill.paradigm_handler.needs_epoched_data and preprocessed_data.ndim == 2:
            preprocessed_data = _auto._epoch_for_features(
                preprocessed_data, events, skill.sfreq
            )

        # ---- Extract features ----
        extractors = skill.paradigm_handler.get_feature_extractors(
            modality=skill.modality, sfreq=skill.sfreq
        )

        feature_matrix, feature_names = _auto._run_extractors(
            preprocessed_data, extractors
        )

        # ---- Build report ----
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
            step_timings=step_timing,
            feature_names=feature_names,
            feature_matrix=feature_matrix,
            knowledge_entry=skill.get_knowledge(),
            total_time_s=sum(step_timing.values()),
        )

        knowledge = skill.get_knowledge() or {}
        references = skill.knowledge_graph.get_references(
            skill.modality, skill.paradigm
        )
        code_template = self._build_combined_template(pipeline, skill)
        explanations = self._build_narrative_text(pipeline, skill)

        return {
            "steps": steps_info,
            "explanations": explanations,
            "code_templates": code_template,
            "knowledge": knowledge,
            "references": references,
            "feature_matrix": feature_matrix,
            "feature_names": feature_names,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Narrative helpers
    # ------------------------------------------------------------------

    def _collect_step_info(self, pipeline) -> List[Dict[str, Any]]:
        """Collect description and code template from each pipeline step."""
        info = []
        for i, step in enumerate(pipeline.steps, start=1):
            info.append(
                {
                    "step_number": i,
                    "name": step.name,
                    "description": step.description,
                    "code_template": step.code_template,
                    "params": step.params,
                }
            )
        return info

    def _print_narrative(self, pipeline, skill) -> None:
        """Print a colourful teaching narrative to stdout."""
        skill_header = (
            f"\n{'=' * 70}\n"
            f" OpenBrainSkill — Teaching Mode\n"
            f" Modality : {skill.modality.upper()}\n"
            f" Paradigm : {skill.paradigm.replace('_', ' ').title()}\n"
            f"{'=' * 70}"
        )
        print(skill_header)

        description = skill.describe()
        if description:
            print(f"\n📖 Paradigm Overview\n{'-' * 40}")
            print(description)

        kg_steps = skill.get_preprocessing_steps()
        if kg_steps:
            print(f"\n📚 Recommended Preprocessing (Knowledge Graph)\n{'-' * 40}")
            for step in kg_steps:
                name = step.get("step", step.get("name", "unknown"))
                rationale = step.get("rationale", "")
                print(f"  • {name}: {rationale}")

        print(f"\n🔧 Active Pipeline Steps\n{'-' * 40}")
        for i, step in enumerate(pipeline.steps, 1):
            print(f"\n  Step {i}: {step.name}")
            if step.description:
                # Word-wrap description
                words = step.description.split()
                line, lines = "", []
                for word in words:
                    if len(line) + len(word) + 1 > 66:
                        lines.append(line)
                        line = word
                    else:
                        line = f"{line} {word}".strip()
                if line:
                    lines.append(line)
                for l in lines:
                    print(f"    {l}")

            if step.code_template:
                print(f"\n  💻 Code template:")
                for code_line in step.code_template.split("\n"):
                    print(f"      {code_line}")

        refs = skill.knowledge_graph.get_references(skill.modality, skill.paradigm)
        if refs:
            print(f"\n📄 Key References\n{'-' * 40}")
            for ref in refs:
                print(f"  [{refs.index(ref) + 1}] {ref}")

        print(f"\n{'=' * 70}\n")

    def _build_narrative_text(self, pipeline, skill) -> str:
        """Build a multi-line text narrative of the pipeline."""
        lines = [
            f"OpenBrainSkill — Teaching Mode",
            f"Modality: {skill.modality.upper()} | Paradigm: {skill.paradigm}",
            "",
        ]
        desc = skill.describe()
        if desc:
            lines += ["Paradigm Overview:", desc, ""]

        lines.append("Pipeline Steps:")
        for i, step in enumerate(pipeline.steps, 1):
            lines.append(f"  {i}. {step.name}: {step.description}")

        refs = skill.knowledge_graph.get_references(skill.modality, skill.paradigm)
        if refs:
            lines.append("")
            lines.append("References:")
            for ref in refs:
                lines.append(f"  - {ref}")

        return "\n".join(lines)

    def _build_combined_template(self, pipeline, skill) -> str:
        """Build a single runnable Python code template for the full pipeline."""
        lines = [
            "# ==============================================================",
            f"# OpenBrainSkill — {skill.modality.upper()} / {skill.paradigm}",
            "# Auto-generated code template",
            "# ==============================================================",
            "",
            "import numpy as np",
            "from openbrain_skill import BrainSkill",
            "",
            f"skill = BrainSkill(",
            f"    modality={skill.modality!r},",
            f"    paradigm={skill.paradigm!r},",
            f"    sfreq={skill.sfreq},",
            f"    n_channels={skill.n_channels},",
            ")",
            "",
            "# Generate or load your data",
            "data = skill.modality_handler.generate_synthetic_data(seed=42)",
            "events = skill.paradigm_handler.generate_events(",
            "    n_samples=data.shape[-1], sfreq=skill.sfreq",
            ")",
            "",
            "# ---- Preprocessing steps ----",
        ]

        for step in pipeline.steps:
            lines.append(f"# Step: {step.name}")
            if step.code_template:
                for code_line in step.code_template.split("\n"):
                    lines.append(code_line)
            lines.append("")

        lines += [
            "# ---- Feature extraction ----",
            "results = skill.run_auto(data, events=events)",
            "feature_matrix = results['feature_matrix']",
            "print('Feature matrix shape:', feature_matrix.shape)",
        ]

        return "\n".join(lines)
