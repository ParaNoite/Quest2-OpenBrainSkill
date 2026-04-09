"""Registry for modalities and paradigms in OpenBrainSkill."""
from __future__ import annotations

from typing import Dict, Type


class Registry:
    """Central registry that maps modality and paradigm names to their handler classes.

    Allows dynamic registration and retrieval of modality and paradigm
    handlers, enabling the framework to be extended without modifying
    existing code.

    Attributes:
        _modalities: Mapping of modality name to modality class.
        _paradigms: Mapping of paradigm name to paradigm class.
    """

    def __init__(self) -> None:
        self._modalities: Dict[str, type] = {}
        self._paradigms: Dict[str, type] = {}

    # ------------------------------------------------------------------
    # Modality registration
    # ------------------------------------------------------------------

    def register_modality(self, name: str, cls: type) -> None:
        """Register a modality handler class under *name*.

        Args:
            name: Canonical lower-case modality name (e.g. ``"eeg"``).
            cls: Modality handler class.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._modalities:
            raise ValueError(f"Modality '{name}' is already registered.")
        self._modalities[name] = cls

    def get_modality(self, name: str) -> type:
        """Return the modality class registered under *name*.

        Args:
            name: Canonical modality name.

        Returns:
            The registered modality class.

        Raises:
            KeyError: If no modality with that name has been registered.
        """
        if name not in self._modalities:
            raise KeyError(
                f"Unknown modality '{name}'. "
                f"Available: {list(self._modalities)}"
            )
        return self._modalities[name]

    def list_modalities(self) -> list[str]:
        """Return a list of all registered modality names."""
        return list(self._modalities)

    # ------------------------------------------------------------------
    # Paradigm registration
    # ------------------------------------------------------------------

    def register_paradigm(self, name: str, cls: type) -> None:
        """Register a paradigm handler class under *name*.

        Args:
            name: Canonical lower-case paradigm name (e.g. ``"motor_imagery"``).
            cls: Paradigm handler class.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._paradigms:
            raise ValueError(f"Paradigm '{name}' is already registered.")
        self._paradigms[name] = cls

    def get_paradigm(self, name: str) -> type:
        """Return the paradigm class registered under *name*.

        Args:
            name: Canonical paradigm name.

        Returns:
            The registered paradigm class.

        Raises:
            KeyError: If no paradigm with that name has been registered.
        """
        if name not in self._paradigms:
            raise KeyError(
                f"Unknown paradigm '{name}'. "
                f"Available: {list(self._paradigms)}"
            )
        return self._paradigms[name]

    def list_paradigms(self) -> list[str]:
        """Return a list of all registered paradigm names."""
        return list(self._paradigms)


# ---------------------------------------------------------------------------
# Module-level singleton registry populated with built-in handlers
# ---------------------------------------------------------------------------

_default_registry: Registry | None = None


def get_default_registry() -> Registry:
    """Return (and lazily create) the module-level default registry.

    Built-in modalities and paradigms are registered on first access.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = _build_default_registry()
    return _default_registry


def _build_default_registry() -> Registry:
    """Instantiate a Registry pre-populated with built-in handlers."""
    from openbrain_skill.modalities.eeg import EEGModality
    from openbrain_skill.modalities.fnirs import FNIRSModality
    from openbrain_skill.modalities.ecog import ECoGModality
    from openbrain_skill.paradigms.motor_imagery import MotorImageryParadigm
    from openbrain_skill.paradigms.p300 import P300Paradigm
    from openbrain_skill.paradigms.ssvep import SSVEPParadigm
    from openbrain_skill.paradigms.resting_state import RestingStateParadigm

    registry = Registry()

    # Modalities
    registry.register_modality("eeg", EEGModality)
    registry.register_modality("fnirs", FNIRSModality)
    registry.register_modality("ecog", ECoGModality)

    # Paradigms
    registry.register_paradigm("motor_imagery", MotorImageryParadigm)
    registry.register_paradigm("p300", P300Paradigm)
    registry.register_paradigm("ssvep", SSVEPParadigm)
    registry.register_paradigm("resting_state", RestingStateParadigm)

    return registry
