"""
OpenBrainSkill: Multi-modal brain signal processing skill framework.

A unified framework for EEG, fNIRS, and ECoG signal preprocessing and
feature extraction, supporting both automatic and teaching modes.

Modes:
    - auto_mode: Automatically execute the full pipeline and output feature
      matrices and reports.
    - teaching_mode: Provide interactive step-by-step explanations and
      reusable code templates.
"""

from openbrain_skill.core.pipeline import Pipeline
from openbrain_skill.core.skill import BrainSkill
from openbrain_skill.core.knowledge_graph import KnowledgeGraph
from openbrain_skill.core.registry import Registry
from openbrain_skill.modes.auto_mode import AutoMode
from openbrain_skill.modes.teaching_mode import TeachingMode

__version__ = "0.1.0"
__all__ = [
    "Pipeline",
    "BrainSkill",
    "KnowledgeGraph",
    "Registry",
    "AutoMode",
    "TeachingMode",
]
