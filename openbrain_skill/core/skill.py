"""Base BrainSkill class — the top-level orchestrator for OpenBrainSkill."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from openbrain_skill.core.knowledge_graph import KnowledgeGraph
from openbrain_skill.core.registry import get_default_registry, Registry


class BrainSkill:
    """Top-level orchestrator for the OpenBrainSkill framework.

    Instantiates a full preprocessing + feature extraction pipeline for a
    given modality / paradigm combination and delegates execution to the
    requested operating mode.

    Args:
        modality: Signal modality string (e.g. ``"eeg"``).
        paradigm: Experimental paradigm string (e.g. ``"motor_imagery"``).
        sfreq: Sampling frequency of the raw data (Hz).
        n_channels: Number of recording channels.
        registry: Optional custom :class:`Registry`. Defaults to the
            built-in registry.
        knowledge_graph: Optional custom :class:`KnowledgeGraph`.

    Raises:
        KeyError: If *modality* or *paradigm* is not registered.
    """

    def __init__(
        self,
        modality: str,
        paradigm: str,
        sfreq: float = 256.0,
        n_channels: int = 64,
        registry: Optional[Registry] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ) -> None:
        self.modality = modality.lower()
        self.paradigm = paradigm.lower()
        self.sfreq = sfreq
        self.n_channels = n_channels

        self._registry = registry or get_default_registry()
        self._kg = knowledge_graph or KnowledgeGraph()

        # Instantiate modality and paradigm handlers
        modality_cls = self._registry.get_modality(self.modality)
        self._modality_handler = modality_cls(sfreq=sfreq, n_channels=n_channels)

        paradigm_cls = self._registry.get_paradigm(self.paradigm)
        self._paradigm_handler = paradigm_cls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_auto(
        self,
        data: Optional[np.ndarray] = None,
        *,
        channel_names: Optional[list] = None,
        events: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full pipeline in automatic mode.

        Args:
            data: Raw signal array of shape ``(n_channels, n_samples)``.
                If *None*, synthetic data is generated automatically.
            channel_names: Optional list of channel name strings.
            events: Optional event array of shape ``(n_events, 3)``
                containing sample indices and event codes.
            verbose: If *True*, print progress to stdout.

        Returns:
            Dict with keys ``"feature_matrix"``, ``"report"``, and
            ``"metadata"``.
        """
        from openbrain_skill.modes.auto_mode import AutoMode

        mode = AutoMode(skill=self, verbose=verbose)
        return mode.run(data, channel_names=channel_names, events=events)

    def run_teaching(
        self,
        data: Optional[np.ndarray] = None,
        *,
        channel_names: Optional[list] = None,
        events: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline in teaching mode.

        Args:
            data: Optional raw signal array. When *None*, synthetic data are
                generated automatically.
            channel_names: Optional channel name strings.
            events: Optional event array.

        Returns:
            Dict with keys ``"steps"``, ``"explanations"``, ``"code_templates"``,
            ``"feature_matrix"``, and ``"report"``.
        """
        from openbrain_skill.modes.teaching_mode import TeachingMode

        mode = TeachingMode(skill=self)
        return mode.run(data, channel_names=channel_names, events=events)

    # ------------------------------------------------------------------
    # Knowledge graph helpers
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of the modality–paradigm pair."""
        return self._kg.get_description(self.modality, self.paradigm)

    def get_knowledge(self) -> Optional[Dict[str, Any]]:
        """Return the full knowledge entry for this modality–paradigm pair."""
        return self._kg.query(self.modality, self.paradigm)

    def get_preprocessing_steps(self):
        """Return the recommended preprocessing steps from the knowledge graph."""
        return self._kg.get_preprocessing_steps(self.modality, self.paradigm)

    def get_recommended_features(self):
        """Return the recommended features from the knowledge graph."""
        return self._kg.get_features(self.modality, self.paradigm)

    # ------------------------------------------------------------------
    # Internal helpers exposed to modes
    # ------------------------------------------------------------------

    @property
    def modality_handler(self):
        """The instantiated modality handler."""
        return self._modality_handler

    @property
    def paradigm_handler(self):
        """The instantiated paradigm handler."""
        return self._paradigm_handler

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """The knowledge graph used by this skill."""
        return self._kg

    def __repr__(self) -> str:
        return (
            f"BrainSkill(modality={self.modality!r}, paradigm={self.paradigm!r}, "
            f"sfreq={self.sfreq}, n_channels={self.n_channels})"
        )
