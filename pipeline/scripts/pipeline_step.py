"""Abstract base class for all pipeline steps."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any


class PipelineStep(ABC):
    """
    Abstract base class for NLP pipeline steps.

    Each step encapsulates:
    - Processing logic for one transformation stage
    - Input validation
    - Structured logging and timing
    """

    def __init__(self, step_number: int, name: str, description: str):
        """
        Initialize pipeline step.

        Args:
            step_number: Sequential position in the pipeline (0-9)
            name: Human-readable step name
            description: Brief description of the transformation
        """
        self.step_number = step_number
        self.name = name
        self.description = description
        self._logger = logging.getLogger(f"pipeline.step_{step_number}")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Execute the step transformation.

        Args:
            input_data: Output from the previous step

        Returns:
            Transformed data for the next step
        """

    @abstractmethod
    def validate(self, output_data: Any) -> bool:
        """
        Validate the step output before passing downstream.

        Args:
            output_data: Output produced by process()

        Returns:
            True if output is valid, False otherwise
        """

    def run(self, input_data: Any) -> Any:
        """
        Execute process() with timing and validation.

        Args:
            input_data: Input data for this step

        Returns:
            Validated output data

        Raises:
            ValueError: If output validation fails
        """
        start = time.perf_counter()
        self._logger.info(f"Step {self.step_number} [{self.name}] started")
        output = self.process(input_data)
        if not self.validate(output):
            raise ValueError(f"Step {self.step_number} [{self.name}] validation failed")
        elapsed = time.perf_counter() - start
        self._logger.info(f"Step {self.step_number} [{self.name}] completed in {elapsed:.2f}s")
        return output

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(step={self.step_number}, name='{self.name}')"
