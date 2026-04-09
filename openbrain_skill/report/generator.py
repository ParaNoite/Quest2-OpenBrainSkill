"""Report generation for OpenBrainSkill pipeline runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


class ReportGenerator:
    """Generate structured reports for OpenBrainSkill pipeline runs.

    Reports contain:
    - Run metadata (timestamp, modality, paradigm, sfreq, etc.)
    - Pipeline step summary (step names, timings)
    - Feature statistics (mean, std, min, max per feature)
    - Knowledge graph highlights (recommended steps vs. executed steps)
    - Literature references
    """

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        modality: str,
        paradigm: str,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        channel_names: List[str],
        n_events: int,
        pipeline_steps: List[str],
        step_timings: Dict[str, float],
        feature_names: List[str],
        feature_matrix: np.ndarray,
        knowledge_entry: Optional[Dict[str, Any]] = None,
        total_time_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Create a comprehensive report dict.

        Args:
            modality: Signal modality name.
            paradigm: Experimental paradigm name.
            n_channels: Number of recording channels.
            n_samples: Number of time samples in the raw data.
            sfreq: Sampling frequency (Hz).
            channel_names: List of channel name strings.
            n_events: Number of events/trials.
            pipeline_steps: Names of executed preprocessing steps.
            step_timings: Dict mapping step name → wall-clock time (s).
            feature_names: List of feature column names.
            feature_matrix: Extracted feature matrix as np.ndarray.
            knowledge_entry: Optional knowledge graph entry dict.
            total_time_s: Total pipeline wall-clock time (s).

        Returns:
            Structured report dict.
        """
        duration_s = n_samples / sfreq if sfreq > 0 else 0.0

        feature_stats = self._compute_feature_stats(feature_matrix, feature_names)
        knowledge_summary = self._summarise_knowledge(knowledge_entry, pipeline_steps)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "framework": "OpenBrainSkill",
            "version": "0.1.0",
            "metadata": {
                "modality": modality,
                "paradigm": paradigm,
                "sfreq_hz": sfreq,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "duration_s": round(duration_s, 3),
                "n_events": n_events,
            },
            "pipeline": {
                "steps": pipeline_steps,
                "step_timings_s": {k: round(v, 6) for k, v in step_timings.items()},
                "total_preprocessing_s": round(
                    sum(step_timings.values()), 6
                ),
            },
            "features": {
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "output_shape": list(feature_matrix.shape),
                "statistics": feature_stats,
            },
            "knowledge_graph": knowledge_summary,
            "total_time_s": round(total_time_s, 6),
        }

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_feature_stats(
        self, feature_matrix: np.ndarray, feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Compute summary statistics for each feature column."""
        if feature_matrix.size == 0:
            return []

        stats = []
        data = feature_matrix.reshape(-1, feature_matrix.shape[-1])

        for i, name in enumerate(feature_names):
            if i >= data.shape[-1]:
                break
            col = data[:, i]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                stats.append(
                    {"name": name, "mean": None, "std": None, "min": None, "max": None}
                )
            else:
                stats.append(
                    {
                        "name": name,
                        "mean": float(round(float(finite.mean()), 6)),
                        "std": float(round(float(finite.std()), 6)),
                        "min": float(round(float(finite.min()), 6)),
                        "max": float(round(float(finite.max()), 6)),
                    }
                )
        return stats

    def _summarise_knowledge(
        self,
        knowledge_entry: Optional[Dict[str, Any]],
        executed_steps: List[str],
    ) -> Dict[str, Any]:
        """Summarise knowledge graph alignment with the executed pipeline."""
        if knowledge_entry is None:
            return {"available": False}

        recommended = [
            step.get("step", step.get("name", "unknown"))
            for step in knowledge_entry.get("preprocessing", [])
        ]
        recommended_features = [
            feat.get("name", "unknown")
            for feat in knowledge_entry.get("features", [])
        ]
        tools = knowledge_entry.get("recommended_tools", [])
        refs = knowledge_entry.get("references", [])
        description = knowledge_entry.get("description", "")

        coverage = (
            len(set(executed_steps) & set(recommended)) / len(recommended)
            if recommended
            else 1.0
        )

        return {
            "available": True,
            "description": description,
            "recommended_preprocessing": recommended,
            "executed_preprocessing": executed_steps,
            "coverage_fraction": round(coverage, 3),
            "recommended_features": recommended_features,
            "recommended_tools": tools,
            "references": refs,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, report: Dict[str, Any], indent: int = 2) -> str:
        """Serialise a report dict to a JSON string.

        Args:
            report: Report dict returned by :meth:`generate`.
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(report, indent=indent, ensure_ascii=False, default=str)

    def to_text(self, report: Dict[str, Any]) -> str:
        """Render a report dict as a human-readable text summary.

        Args:
            report: Report dict.

        Returns:
            Multi-line text string.
        """
        m = report.get("metadata", {})
        p = report.get("pipeline", {})
        f = report.get("features", {})
        kg = report.get("knowledge_graph", {})

        lines = [
            "=" * 70,
            "  OpenBrainSkill — Processing Report",
            "=" * 70,
            f"  Timestamp  : {report.get('timestamp', 'N/A')}",
            f"  Modality   : {m.get('modality', 'N/A').upper()}",
            f"  Paradigm   : {m.get('paradigm', 'N/A').replace('_', ' ').title()}",
            f"  Sfreq      : {m.get('sfreq_hz', 'N/A')} Hz",
            f"  Channels   : {m.get('n_channels', 'N/A')}",
            f"  Duration   : {m.get('duration_s', 'N/A')} s",
            f"  Events     : {m.get('n_events', 'N/A')}",
            "-" * 70,
            "  Pipeline",
            "-" * 70,
        ]

        for i, step in enumerate(p.get("steps", []), 1):
            t = p.get("step_timings_s", {}).get(step, 0.0)
            lines.append(f"    {i:2d}. {step:<35s} ({t:.4f} s)")

        lines += [
            f"  Total preprocessing: {p.get('total_preprocessing_s', 0):.4f} s",
            "-" * 70,
            "  Features",
            "-" * 70,
            f"  Output shape   : {f.get('output_shape', 'N/A')}",
            f"  Feature count  : {f.get('n_features', 0)}",
        ]

        stats = f.get("statistics", [])
        if stats:
            lines.append(
                f"  {'Feature':<30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
            )
            lines.append(f"  {'-'*70}")
            for stat in stats[:20]:  # Cap at 20 rows
                name = stat["name"][:29]
                lines.append(
                    f"  {name:<30s} "
                    f"{self._fmt(stat['mean']):>10s} "
                    f"{self._fmt(stat['std']):>10s} "
                    f"{self._fmt(stat['min']):>10s} "
                    f"{self._fmt(stat['max']):>10s}"
                )
            if len(stats) > 20:
                lines.append(f"  ... ({len(stats) - 20} more features)")

        if kg.get("available"):
            lines += [
                "-" * 70,
                "  Knowledge Graph",
                "-" * 70,
                f"  Coverage: {kg.get('coverage_fraction', 0):.1%} of recommended steps executed.",
                f"  Recommended tools: {', '.join(kg.get('recommended_tools', []))}",
            ]
            refs = kg.get("references", [])
            if refs:
                lines.append("  References:")
                for ref in refs:
                    lines.append(f"    [{refs.index(ref)+1}] {ref}")

        lines += [
            "-" * 70,
            f"  Total time: {report.get('total_time_s', 0):.4f} s",
            "=" * 70,
        ]

        return "\n".join(lines)

    @staticmethod
    def _fmt(val) -> str:
        """Format a numeric value for display."""
        if val is None:
            return "N/A"
        return f"{val:.4f}"
