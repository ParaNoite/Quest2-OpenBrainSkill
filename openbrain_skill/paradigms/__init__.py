"""Paradigms package for OpenBrainSkill."""

from openbrain_skill.paradigms.base import BaseParadigm
from openbrain_skill.paradigms.motor_imagery import MotorImageryParadigm
from openbrain_skill.paradigms.p300 import P300Paradigm
from openbrain_skill.paradigms.ssvep import SSVEPParadigm
from openbrain_skill.paradigms.resting_state import RestingStateParadigm

__all__ = [
    "BaseParadigm",
    "MotorImageryParadigm",
    "P300Paradigm",
    "SSVEPParadigm",
    "RestingStateParadigm",
]
