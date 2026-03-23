"""Step 8: Similarity intensity visualizer across all Súmulas sorted by topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from pipeline_step import PipelineStep
from step_7_search_index import SearchIndexOutput


@dataclass
class SimilarityVisualizationOutput:
    """
    Output of the similarity visualizer step.

    Attributes:
        figures: List of figure objects produced (matplotlib)
        saved_paths: Paths to the saved figure image files
    """

    figures: list[Any]
    saved_paths: list[Path] = field(default_factory=list)


class SimilarityVisualizer(PipelineStep):
    """
    Plot cosine similarity distribution across all Súmulas.

    Produces one chart:
    - A histogram of cosine similarity distribution across all súmulas.

    The figure is saved to output_dir.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize similarity visualizer.

        Args:
            output_dir: Directory for saving figure images; created if absent
        """
        super().__init__(
            step_number=8,
            name="Similarity Visualizer",
            description="Plot similarity intensity across all Súmulas sorted by topic",
        )
        self._output_dir = Path(output_dir) if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_stacked_histograms(
        self, outputs: list[SearchIndexOutput]
    ) -> matplotlib.figure.Figure:
        """
        Build vertically stacked histograms of cosine similarity, one per query.

        All subplots share the same X axis scale. Each subplot shows the
        similarity distribution for one query with the accumulated similarity
        displayed as a right-side twin-axis label.

        Args:
            outputs: List of SearchIndexOutput instances, one per query

        Returns:
            matplotlib Figure with len(outputs) vertically stacked subplots
        """
        n = len(outputs)
        fig, raw_axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(12, 4 * n),
            sharex=True,
        )
        axes_list: list[matplotlib.axes.Axes] = [raw_axes] if n == 1 else list(raw_axes)
        for ax, output in zip(axes_list, outputs):
            similarities = [r.similarity for r in output.results]
            ax.hist(similarities, bins=40, color="steelblue", edgecolor="white")
            ax.set_ylabel("Count")
            ax.set_title(
                f"{output.query[:80]!r}",
                fontsize=10,
                pad=6,
            )
            ax.text(
                0.02, 0.95,
                f"Accumulated Similarity: {output.accumulated_similarity:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12,
                fontweight="bold",
                color="steelblue",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "steelblue", "alpha": 0.85},
            )
        axes_list[-1].set_xlabel("Cosine Similarity")
        fig.suptitle("Similarity Distribution by Query", fontsize=13, y=1.01)
        plt.tight_layout()
        return fig

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

    def process(
        self, input_data: SearchIndexOutput | list[SearchIndexOutput]
    ) -> SimilarityVisualizationOutput:
        """
        Produce stacked cosine similarity histograms for all input queries.

        Accepts a single SearchIndexOutput or a list of outputs, one per query.

        Args:
            input_data: Single SearchIndexOutput or list of SearchIndexOutput instances

        Returns:
            SimilarityVisualizationOutput with the stacked histogram figure and saved path
        """
        outputs = [input_data] if isinstance(input_data, SearchIndexOutput) else input_data
        stacked_fig = self._build_stacked_histograms(outputs)
        saved_paths: list[Path] = []
        stacked_path = self._save_figure(stacked_fig, "similarity_histogram.png")
        if stacked_path:
            saved_paths.append(stacked_path)
        return SimilarityVisualizationOutput(
            figures=[stacked_fig],
            saved_paths=saved_paths,
        )

    def validate(self, output_data: SimilarityVisualizationOutput) -> bool:
        """
        Validate that at least one figure was produced.

        Args:
            output_data: SimilarityVisualizationOutput to validate

        Returns:
            True if figures list is non-empty
        """
        return len(output_data.figures) > 0
