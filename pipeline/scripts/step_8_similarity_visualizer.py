"""Step 8: Similarity intensity visualizer across all Súmulas sorted by topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pipeline_step import PipelineStep
from step_7_search_index import SearchIndexOutput, SearchResult


@dataclass
class SimilarityVisualizationOutput:
    """
    Output of the similarity visualizer step.

    Attributes:
        figures: List of matplotlib Figure objects produced
        saved_paths: Paths to the saved figure image files
    """

    figures: list[matplotlib.figure.Figure]
    saved_paths: list[Path] = field(default_factory=list)


class SimilarityVisualizer(PipelineStep):
    """
    Plot cosine similarity intensity across all Súmulas sorted by topic name.

    Produces two charts:
    - A histogram of cosine similarity distribution across all súmulas.
    - A heatmap matrix where rows are topic groups (aggregated mean similarity)
      and columns represent those same topic groups, allowing cluster intensity
      patterns to be identified across the full dataset.

    Both figures are saved to output_dir.
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
        self, results: list[SearchResult], query: str
    ) -> matplotlib.figure.Figure:
        """
        Build histogram of cosine similarity distribution across all súmulas.

        Args:
            results: SearchResult list covering all súmulas in the corpus
            query: Original query string for the chart title

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
        plt.tight_layout()
        return fig

    def _build_heatmap(
        self, results: list[SearchResult], query: str
    ) -> matplotlib.figure.Figure:
        """
        Build heatmap matrix of mean cosine similarity aggregated by topic group.

        Rows and columns both represent topic groups derived from the composite
        label. Each cell encodes the mean similarity of súmulas belonging to that
        row-topic against the column-topic, enabling cross-topic cluster analysis.
        Groups are sorted alphabetically so contiguous topics align visually.

        Args:
            results: SearchResult list covering all súmulas in the corpus
            query: Original query string for the figure title

        Returns:
            matplotlib Figure with the topic-group heatmap
        """
        topic_scores: dict[str, list[float]] = {}
        for result in results:
            topic_scores.setdefault(result.label, []).append(result.similarity)
        topics = sorted(topic_scores.keys())
        mean_scores = np.array([np.mean(topic_scores[t]) for t in topics])
        matrix = np.outer(mean_scores, mean_scores)
        n_topics = len(topics)
        cell_size = max(0.35, min(0.6, 20.0 / n_topics))
        fig_size = max(8, n_topics * cell_size)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax, label="Mean Cosine Similarity", shrink=0.6, pad=0.02)
        font_size = max(4, min(8, 200 // n_topics))
        ax.set_xticks(np.arange(n_topics))
        ax.set_yticks(np.arange(n_topics))
        ax.set_xticklabels(topics, rotation=90, fontsize=font_size)
        ax.set_yticklabels(topics, fontsize=font_size)
        ax.set_xlabel("Topic Group")
        ax.set_ylabel("Topic Group")
        ax.set_title(
            f"Topic-Group Similarity Heatmap — {query[:70]!r}",
            fontsize=11,
            pad=10,
        )
        plt.tight_layout()
        return fig

    def _save_figure(
        self, fig: matplotlib.figure.Figure, filename: str
    ) -> Optional[Path]:
        """
        Save a figure to the configured output directory.

        Args:
            fig: Figure to save
            filename: Target filename including extension

        Returns:
            Resolved Path where the figure was saved, or None when no output_dir is set
        """
        if not self._output_dir:
            return None
        path = self._output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def process(self, input_data: SearchIndexOutput) -> SimilarityVisualizationOutput:
        """
        Produce the similarity histogram and topic heatmap from search index output.

        Args:
            input_data: SearchIndexOutput with results covering all súmulas

        Returns:
            SimilarityVisualizationOutput with figures and saved paths
        """
        histogram_fig = self._build_histogram(input_data.results, input_data.query)
        heatmap_fig = self._build_heatmap(input_data.results, input_data.query)
        saved_paths: list[Path] = []
        histogram_path = self._save_figure(histogram_fig, "similarity_histogram.png")
        if histogram_path:
            saved_paths.append(histogram_path)
        heatmap_path = self._save_figure(heatmap_fig, "similarity_heatmap.png")
        if heatmap_path:
            saved_paths.append(heatmap_path)
        return SimilarityVisualizationOutput(
            figures=[histogram_fig, heatmap_fig],
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
