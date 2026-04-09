"""Pipeline: ordered sequence of preprocessing and feature extraction steps."""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class PipelineStep:
    """A single named step in the processing pipeline.

    Args:
        name: Human-readable step name.
        func: Callable that takes ``(data, **kwargs)`` and returns the
            transformed data.
        params: Additional keyword arguments forwarded to *func*.
        description: Optional textual description for teaching mode.
        code_template: Optional Python code snippet illustrating the step.
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        code_template: str = "",
    ) -> None:
        self.name = name
        self.func = func
        self.params: Dict[str, Any] = params or {}
        self.description = description
        self.code_template = code_template

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute this step on *data*.

        Args:
            data: Input signal array.

        Returns:
            Transformed signal array.
        """
        return self.func(data, **self.params)

    def __repr__(self) -> str:
        return f"PipelineStep(name={self.name!r})"


class Pipeline:
    """Ordered sequence of :class:`PipelineStep` objects.

    Provides a simple run-all interface and records per-step timing.

    Args:
        steps: Initial list of steps (default: empty).
    """

    def __init__(self, steps: Optional[List[PipelineStep]] = None) -> None:
        self._steps: List[PipelineStep] = list(steps or [])
        self._step_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Append a step to the pipeline.

        Args:
            step: Step to append.

        Returns:
            *self* for chaining.
        """
        self._steps.append(step)
        return self

    def remove_step(self, name: str) -> "Pipeline":
        """Remove the first step with the given name.

        Args:
            name: Step name to remove.

        Returns:
            *self* for chaining.

        Raises:
            KeyError: If no step with *name* is found.
        """
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                return self
        raise KeyError(f"Step '{name}' not found in pipeline.")

    def get_step(self, name: str) -> PipelineStep:
        """Return the first step with the given name.

        Args:
            name: Step name.

        Returns:
            Matching :class:`PipelineStep`.

        Raises:
            KeyError: If not found.
        """
        for step in self._steps:
            if step.name == name:
                return step
        raise KeyError(f"Step '{name}' not found in pipeline.")

    @property
    def steps(self) -> List[PipelineStep]:
        """Ordered list of pipeline steps."""
        return list(self._steps)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        data: np.ndarray,
        *,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute all steps in order.

        Args:
            data: Input signal array of shape ``(n_channels, n_samples)``
                or ``(n_epochs, n_channels, n_samples)``.
            verbose: If *True*, print step names as they execute.

        Returns:
            Tuple of ``(transformed_data, step_info_dict)``.
        """
        self._step_timings = {}
        step_info: Dict[str, Any] = {}
        current_data = data

        for step in self._steps:
            if verbose:
                print(f"  [Pipeline] Running step: {step.name}")
            t0 = time.perf_counter()
            current_data = step.run(current_data)
            elapsed = time.perf_counter() - t0
            self._step_timings[step.name] = elapsed
            step_info[step.name] = {
                "elapsed_s": elapsed,
                "output_shape": (
                    current_data.shape if isinstance(current_data, np.ndarray) else None
                ),
            }

        return current_data, step_info

    @property
    def step_timings(self) -> Dict[str, float]:
        """Per-step wall-clock timings (seconds) from the last :meth:`run`."""
        return dict(self._step_timings)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        names = [s.name for s in self._steps]
        return f"Pipeline(steps={names})"
