"""Shared base class for pipeline steps that produce matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pipeline_step import PipelineStep


class VisualizationStep(PipelineStep):
    """
    Intermediate base class for visualization pipeline steps.

    Provides shared output directory management and figure persistence so
    concrete steps only implement their domain-specific plotting logic.
    """

    def __init__(
        self,
        step_number: int,
        name: str,
        description: str,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize visualization step.

        Args:
            step_number: Sequential position in the pipeline
            name: Human-readable step name
            description: Brief description of the visualization
            output_dir: Directory for saving figure images; created if absent
        """
        super().__init__(step_number=step_number, name=name, description=description)
        self._output_dir = Path(output_dir) if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, fig: Any, filename: str) -> Optional[Path]:
        """
        Save a matplotlib figure to the configured output directory.

        Args:
            fig: matplotlib Figure to save
            filename: Target filename including extension

        Returns:
            Resolved Path where the figure was saved, or None when no output_dir is set
        """
        if not self._output_dir:
            return None
        path = self._output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
