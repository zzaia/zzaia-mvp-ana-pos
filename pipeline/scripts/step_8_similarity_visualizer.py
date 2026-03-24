"""Step 8: Similarity intensity visualizer across all Súmulas sorted by topic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

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

        Subplots are sorted descending by mean similarity. All subplots
        share the same X axis. Bars are colored by legal area label. Vertical
        dashed lines mark min and max similarity. Level 1 descriptive statistics
        are annotated inside each subplot.

        Args:
            outputs: List of SearchIndexOutput instances, one per query

        Returns:
            matplotlib Figure with len(outputs) vertically stacked subplots
        """
        sorted_outputs = sorted(outputs, key=lambda o: o.mean_similarity, reverse=True)
        all_areas: list[str] = sorted({r.area for o in sorted_outputs for r in o.results})
        color_map = plt.colormaps.get_cmap("tab10")
        area_colors: dict[str, Any] = {area: color_map(i / max(len(all_areas), 1)) for i, area in enumerate(all_areas)}
        n = len(sorted_outputs)
        fig, raw_axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(12, 4 * n),
            sharex=True,
        )
        axes_list: list[matplotlib.axes.Axes] = [raw_axes] if n == 1 else list(raw_axes)
        for ax, output in zip(axes_list, sorted_outputs):
            similarities = [r.similarity for r in output.results]
            areas = [r.area for r in output.results]
            unique_areas_in_output = sorted(set(areas))
            bins = np.linspace(min(similarities), max(similarities), 41)
            bottom = np.zeros(len(bins) - 1)
            for area in unique_areas_in_output:
                area_sims = [s for s, a in zip(similarities, areas) if a == area]
                counts, _ = np.histogram(area_sims, bins=bins)
                ax.bar(
                    bins[:-1],
                    counts,
                    width=np.diff(bins),
                    bottom=bottom,
                    color=area_colors[area],
                    label=area,
                    align="edge",
                    edgecolor="none",
                )
                bottom += counts.astype(float)
            ax.axvline(min(similarities), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(max(similarities), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(
                0.02, 0.95,
                (
                    f"n={len(output.results)}  mean={output.mean_similarity:.4f}  median={output.median_similarity:.4f}  max={output.max_similarity:.4f}  min={output.min_similarity:.4f}\n"
                    f"std={output.std_similarity:.4f}  var={output.variance_similarity:.5f}  range={output.range_similarity:.4f}  IQR={output.iqr_similarity:.4f}  CV={output.cv_similarity:.4f}"
                ),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                fontweight="bold",
                color="steelblue",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "steelblue", "alpha": 0.85},
            )
            ax.set_ylabel("Count")
            ax.set_title(f"{output.query[:90]!r}", fontsize=10, pad=6)
            ax.legend(fontsize=7, loc="upper right", ncol=3)
        axes_list[-1].set_xlabel("Cosine Similarity")
        fig.suptitle("Similarity Distribution by Query (sorted by Mean Similarity)", fontsize=13, y=1.01)
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
        self, input_data: list[SearchIndexOutput] | SearchIndexOutput
    ) -> SimilarityVisualizationOutput:
        """
        Produce stacked cosine similarity histograms for all input queries.

        Args:
            input_data: List of SearchIndexOutput instances or a single one

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
