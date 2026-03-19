"""Step 8: Similarity intensity visualizer across all Súmulas sorted by topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pipeline_step import PipelineStep
from step_7_search_index import SearchIndexOutput, SearchResult


@dataclass
class SimilarityVisualizationOutput:
    """
    Output of the similarity visualizer step.

    Attributes:
        figures: List of figure objects produced (matplotlib or plotly)
        saved_paths: Paths to the saved figure image files
    """

    figures: list[Any]
    saved_paths: list[Path] = field(default_factory=list)


class SimilarityVisualizer(PipelineStep):
    """
    Plot cosine similarity intensity across all Súmulas sorted by súmula number.

    Produces two charts:
    - A histogram of cosine similarity distribution across all súmulas.
    - A horizontal heatmap bar chart where each query is a labeled row and
      each column is a súmula in document order; cell color encodes cosine
      similarity from the FAISS search results.

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

    def _build_heatmap(self, outputs: list[SearchIndexOutput]) -> go.Figure:
        """
        Build a horizontal heatmap bar chart of cosine similarity per (query, súmula).

        Each query is rendered as a horizontal colored bar. The X axis lists all
        súmulas in ascending sumula_number order; the Y axis lists truncated query
        strings. Cell color encodes cosine similarity (0.0–1.0) from FAISS results,
        defaulting to 0.0 when a súmula is absent from a query's result set.

        Args:
            outputs: List of SearchIndexOutput, one per query (max five)

        Returns:
            Plotly Figure with interactive horizontal heatmap bars
        """
        sumula_index: dict[int, SearchResult] = {}
        for output in outputs:
            for result in output.results:
                sumula_index.setdefault(result.sumula_number, result)
        sorted_numbers = sorted(sumula_index.keys())
        x_labels = [sumula_index[s].label for s in sorted_numbers]
        n_queries = len(outputs)
        n_sumulas = len(sorted_numbers)
        matrix = np.zeros((n_queries, n_sumulas), dtype=float)
        for row, output in enumerate(outputs):
            result_map = {r.sumula_number: r.similarity for r in output.results}
            for col, sumula_number in enumerate(sorted_numbers):
                matrix[row, col] = result_map.get(sumula_number, 0.0)
        y_labels = [
            q[:60] + "..." if len(q) > 60 else q
            for q in (o.query for o in outputs)
        ]
        row_height = 80
        fig_height = max(200, n_queries * row_height + 120)
        fig_width = max(900, n_sumulas * 14)
        fig = px.imshow(
            matrix,
            x=x_labels,
            y=y_labels,
            color_continuous_scale="YlOrRd",
            zmin=0.0,
            zmax=1.0,
            aspect="auto",
            labels={"color": "Cosine Similarity", "x": "Súmula", "y": "Query"},
        )
        fig.update_layout(
            title={"text": "Query × Súmula Cosine Similarity", "font": {"size": 14}},
            xaxis={"tickangle": -90, "tickfont": {"size": 9}},
            yaxis={"tickfont": {"size": 11}, "autorange": "reversed"},
            coloraxis_colorbar={"title": "Cosine Similarity", "thickness": 16},
            width=fig_width,
            height=fig_height,
            margin={"l": 20, "r": 20, "t": 50, "b": 120},
        )
        return fig

    def _save_figure(self, fig: Any, filename: str) -> Optional[Path]:
        """
        Save a figure to the configured output directory.

        Supports matplotlib Figure (saved as PNG via savefig) and Plotly Figure
        (saved as PNG via write_image; requires kaleido).

        Args:
            fig: matplotlib or Plotly Figure to save
            filename: Target filename including extension

        Returns:
            Resolved Path where the figure was saved, or None when no output_dir is set
        """
        if not self._output_dir:
            return None
        path = self._output_dir / filename
        if isinstance(fig, go.Figure):
            fig.write_image(str(path), scale=2)
            return path
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def process(
        self, input_data: SearchIndexOutput | list[SearchIndexOutput]
    ) -> SimilarityVisualizationOutput:
        """
        Produce the similarity histogram and query-by-súmula horizontal heatmap.

        Accepts a single SearchIndexOutput or a list of up to five outputs,
        one per query. The histogram is built from the first output only.

        Args:
            input_data: Single SearchIndexOutput or list of SearchIndexOutput instances

        Returns:
            SimilarityVisualizationOutput with figures and saved paths
        """
        outputs = [input_data] if isinstance(input_data, SearchIndexOutput) else input_data
        first = outputs[0]
        histogram_fig = self._build_histogram(first.results, first.query)
        heatmap_fig = self._build_heatmap(outputs)
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
