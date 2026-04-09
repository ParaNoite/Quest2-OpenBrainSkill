"""Modalities package for OpenBrainSkill."""

from openbrain_skill.modalities.base import BaseModality
from openbrain_skill.modalities.eeg import EEGModality
from openbrain_skill.modalities.fnirs import FNIRSModality
from openbrain_skill.modalities.ecog import ECoGModality

__all__ = ["BaseModality", "EEGModality", "FNIRSModality", "ECoGModality"]
