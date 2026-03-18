"""Step 8: Similarity distribution visualizer for Súmula semantic search results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from step_7_search_index import SearchResult, SumulaSearchIndex


@dataclass
class SimilarityVisualizationOutput:
    """
    Output of the similarity visualizer.

    Attributes:
        figures: List of matplotlib Figure objects produced
        saved_paths: Paths to the saved figure image files
    """

    figures: list[matplotlib.figure.Figure]
    saved_paths: list[Path] = field(default_factory=list)


class SimilarityVisualizer:
    """
    Plot cosine similarity distribution over Súmulas for a given query.

    Produces a ranked horizontal bar chart and a similarity histogram from
    FAISS search results. Both figures are saved to output_dir when provided.
    """

    def __init__(
        self,
        search_index: SumulaSearchIndex,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize similarity visualizer.

        Args:
            search_index: Populated SumulaSearchIndex to query
            output_dir: Directory for saving figure images; created if absent
        """
        self._search_index = search_index
        self._output_dir = Path(output_dir) if output_dir else None
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_bar_chart(self, results: list[SearchResult], query: str) -> matplotlib.figure.Figure:
        """
        Build a horizontal bar chart of top results ranked by similarity.

        Args:
            results: Ordered SearchResult list (highest similarity first)
            query: Original query string for chart title

        Returns:
            matplotlib Figure with the bar chart
        """
        labels = [r.label[:30] for r in results]
        similarities = [r.similarity for r in results]
        colors = plt.cm.viridis([s for s in similarities])
        fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.4)))
        ax.barh(range(len(labels)), similarities[::-1], color=colors[::-1])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("Cosine Similarity")
        ax.set_xlim(0, 1)
        ax.set_title(f"Similarity Distribution — {query[:60]!r}")
        plt.tight_layout()
        return fig

    def _build_histogram(self, results: list[SearchResult]) -> matplotlib.figure.Figure:
        """
        Build a histogram of similarity score distribution.

        Args:
            results: SearchResult list to aggregate similarity values from

        Returns:
            matplotlib Figure with the histogram
        """
        similarities = [r.similarity for r in results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(similarities, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Query Similarity Distribution")
        plt.tight_layout()
        return fig

    def _save_figure(self, fig: matplotlib.figure.Figure, filename: str) -> Optional[Path]:
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

    def visualize(self, query: str, top_k: int = 30) -> SimilarityVisualizationOutput:
        """
        Run search and produce both visualization figures for the given query.

        Args:
            query: Portuguese legal text to search for
            top_k: Number of top results to visualize

        Returns:
            SimilarityVisualizationOutput with figures and saved paths
        """
        results = self._search_index.search(query, top_k=top_k)
        bar_fig = self._build_bar_chart(results, query)
        hist_fig = self._build_histogram(results)
        saved_paths: list[Path] = []
        bar_path = self._save_figure(bar_fig, "similarity_ranking.png")
        hist_path = self._save_figure(hist_fig, "similarity_histogram.png")
        if bar_path:
            saved_paths.append(bar_path)
        if hist_path:
            saved_paths.append(hist_path)
        return SimilarityVisualizationOutput(
            figures=[bar_fig, hist_fig],
            saved_paths=saved_paths,
        )
