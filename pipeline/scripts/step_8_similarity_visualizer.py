"""Step 8: Similarity intensity visualizer across all Súmulas sorted by topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.figure
import matplotlib.pyplot as plt

from pipeline_step import PipelineStep
from step_7_search_index import SearchIndexOutput, SearchResult


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

    def _build_histogram(
        self, results: list[SearchResult], query: str, accumulated_similarity: float
    ) -> matplotlib.figure.Figure:
        """
        Build histogram of cosine similarity distribution across all súmulas.

        Args:
            results: SearchResult list covering all súmulas in the corpus
            query: Original query string for the chart title
            accumulated_similarity: Sum of cosine similarities across all results

        Returns:
            matplotlib Figure with the similarity histogram
        """
        similarities = [r.similarity for r in results]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(similarities, bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Similarity Distribution — {query[:70]!r}",
            fontsize=11,
            pad=10,
        )
        ax2 = ax.twinx()
        ax2.set_ylabel(
            f"Accumulated Similarity: {accumulated_similarity:.2f}",
            fontsize=10,
            color="steelblue",
            labelpad=12,
        )
        ax2.set_yticks([])
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
        Produce the cosine similarity histogram.

        Accepts a single SearchIndexOutput or a list of outputs.
        The histogram is built from the first output only.

        Args:
            input_data: Single SearchIndexOutput or list of SearchIndexOutput instances

        Returns:
            SimilarityVisualizationOutput with figures and saved paths
        """
        outputs = [input_data] if isinstance(input_data, SearchIndexOutput) else input_data
        first = outputs[0]
        histogram_fig = self._build_histogram(first.results, first.query, first.accumulated_similarity)
        saved_paths: list[Path] = []
        histogram_path = self._save_figure(histogram_fig, "similarity_histogram.png")
        if histogram_path:
            saved_paths.append(histogram_path)
        return SimilarityVisualizationOutput(
            figures=[histogram_fig],
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
