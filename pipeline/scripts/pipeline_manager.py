"""Pipeline manager that orchestrates sequential execution with checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pipeline_step import PipelineStep


class PipelineManager:
    """
    Orchestrate the sequential execution of NLP pipeline steps.

    Supports checkpoint-based persistence so that expensive steps (e.g.,
    BERTimbau inference) can be skipped on re-runs when their output is
    already cached on disk.
    """

    def __init__(self, checkpoint_dir: Path, steps: list[PipelineStep]):
        """
        Initialize pipeline manager.

        Args:
            checkpoint_dir: Directory for storing step outputs
            steps: Ordered list of PipelineStep instances (steps 0–9)
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._steps = {step.step_number: step for step in steps}
        self._logger = logging.getLogger("pipeline.manager")

    def run(self, initial_input: Any, start_step: int = 0) -> dict[int, Any]:
        """
        Execute all steps sequentially from start_step onward.

        Args:
            initial_input: Input data for the first step to execute
            start_step: Step number to begin from (allows resuming)

        Returns:
            Dictionary mapping step number to its output
        """
        results: dict[int, Any] = {}
        current_input = initial_input
        sorted_steps = sorted(self._steps.keys())
        for step_num in sorted_steps:
            if step_num < start_step:
                cached = self._load_checkpoint(step_num)
                if cached is not None:
                    results[step_num] = cached
                    current_input = cached
                continue
            cached = self._load_checkpoint(step_num)
            if cached is not None:
                self._logger.info(f"Step {step_num} loaded from checkpoint")
                results[step_num] = cached
                current_input = cached
                continue
            step = self._steps[step_num]
            output = step.run(current_input)
            self._save_checkpoint(step_num, output)
            results[step_num] = output
            current_input = output
        return results

    def run_step(self, step_number: int, input_data: Any) -> Any:
        """
        Execute a single step by its number, loading from checkpoint when available.

        Args:
            step_number: The step to execute
            input_data: Input for the specified step (ignored when checkpoint exists)

        Returns:
            Step output
        """
        cached = self._load_checkpoint(step_number)
        if cached is not None:
            self._logger.info(f"Step {step_number} loaded from checkpoint")
            return cached
        step = self._steps[step_number]
        output = step.run(input_data)
        self._save_checkpoint(step_number, output)
        return output

    def _checkpoint_path(self, step_number: int) -> Path:
        """
        Resolve the checkpoint file path for a step.

        Args:
            step_number: Pipeline step number

        Returns:
            Path to the step .pkl checkpoint file
        """
        return self._checkpoint_dir / f"step_{step_number:02d}.pkl"

    def _save_checkpoint(self, step_number: int, data: Any) -> None:
        """
        Persist step output to disk.

        Args:
            step_number: Step whose output to save
            data: Step output dataclass instance
        """
        import pickle
        pkl_path = self._checkpoint_path(step_number)
        with open(pkl_path, "wb") as fp:
            pickle.dump(data, fp)
        self._logger.info(f"Checkpoint saved: {pkl_path}")

    def _load_checkpoint(self, step_number: int) -> Optional[Any]:
        """
        Load a previously saved step output from disk.

        Args:
            step_number: Step whose checkpoint to load

        Returns:
            Deserialized output or None if not found
        """
        import pickle
        pkl_path = self._checkpoint_path(step_number)
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as fp:
            return pickle.load(fp)

    def clear_checkpoints(self, from_step: int = 0) -> None:
        """
        Delete checkpoint files starting from a given step.

        Args:
            from_step: First step number whose checkpoint to delete
        """
        for step_num in sorted(self._steps.keys()):
            if step_num >= from_step:
                pkl_path = self._checkpoint_path(step_num)
                if pkl_path.exists():
                    pkl_path.unlink()
                    self._logger.info(f"Cleared checkpoint: {pkl_path}")

    def checkpoint_status(self) -> dict[int, bool]:
        """
        Report which steps have existing checkpoints.

        Returns:
            Dictionary mapping step number to checkpoint existence bool
        """
        return {
            step_num: self._checkpoint_path(step_num).exists()
            for step_num in sorted(self._steps.keys())
        }
