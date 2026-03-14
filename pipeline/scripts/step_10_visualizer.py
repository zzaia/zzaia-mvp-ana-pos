"""Step 10: Visualization of UMAP scatter, frequency bar chart, and role heatmap."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from pipeline_step import PipelineStep
from step_8_hdbscan_clusterer import ClusteringOutput
from step_9_statistics_generator import StatisticsOutput

_ROLE_COLORS = {
    "Fact": "#4C72B0",
    "Argument": "#DD8452",
    "Ruling": "#55A868",
    "Precedent": "#C44E52",
    "Procedural": "#8172B3",
    "Uncertain": "#937860",
}


@dataclass
class VisualizationInput:
    """
    Combined input for the visualization step.

    Attributes:
        clustering_output: ClusteringOutput from step 8
        statistics_output: StatisticsOutput from step 9
        output_dir: Directory where figures will be saved (optional)
    """

    clustering_output: ClusteringOutput
    statistics_output: StatisticsOutput
    output_dir: Optional[Path] = None


@dataclass
class VisualizationOutput:
    """
    Output of the visualization step.

    Attributes:
        figures: List of matplotlib Figure objects produced
        saved_paths: Paths to saved figure files (empty if output_dir was None)
    """

    figures: list[matplotlib.figure.Figure] = field(default_factory=list)
    saved_paths: list[Path] = field(default_factory=list)


class Visualizer(PipelineStep):
    """
    Generate three key visualizations for the semantic frequency analysis.

    Produces:
    1. UMAP scatter plot coloured by rhetorical role and cluster ID
    2. Frequency bar chart of sentence counts per cluster
    3. Role distribution heatmap across clusters
    """

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Default matplotlib figure size (width, height)
        """
        super().__init__(
            step_number=10,
            name="Visualizer",
            description="Generate UMAP scatter, frequency chart, and role heatmap",
        )
        self._figsize = figsize

    def process(self, input_data: VisualizationInput) -> VisualizationOutput:
        """
        Produce all three figures.

        Args:
            input_data: VisualizationInput with clustering and statistics outputs

        Returns:
            VisualizationOutput with figures and optional saved paths
        """
        figures: list[matplotlib.figure.Figure] = []
        saved: list[Path] = []
        fig_scatter = self._umap_scatter(input_data.clustering_output)
        figures.append(fig_scatter)
        fig_freq = self._frequency_bar(input_data.statistics_output)
        figures.append(fig_freq)
        fig_heat = self._role_heatmap(input_data.statistics_output)
        figures.append(fig_heat)
        if input_data.output_dir is not None:
            output_dir = Path(input_data.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            names = ["umap_scatter.png", "frequency_bar.png", "role_heatmap.png"]
            for fig, name in zip(figures, names):
                path = output_dir / name
                fig.savefig(path, dpi=150, bbox_inches="tight")
                saved.append(path)
        return VisualizationOutput(figures=figures, saved_paths=saved)

    def _umap_scatter(self, clustering: ClusteringOutput) -> matplotlib.figure.Figure:
        """
        Plot 2D UMAP projection (first two components) coloured by role.

        Args:
            clustering: ClusteringOutput with reduced_vector per sentence

        Returns:
            matplotlib Figure
        """
        sentences = clustering.clustered_sentences
        fig, ax = plt.subplots(figsize=self._figsize)
        if not sentences:
            ax.text(0.5, 0.5, "No sentences to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("UMAP Projection by Rhetorical Role")
            plt.tight_layout()
            return fig
        roles = sorted(set(s.role for s in sentences))
        for role in roles:
            subset = [s for s in sentences if s.role == role]
            xs = [s.reduced_vector[0] for s in subset]
            ys = [s.reduced_vector[1] for s in subset]
            color = _ROLE_COLORS.get(role, "#999999")
            ax.scatter(xs, ys, label=role, color=color, alpha=0.6, s=20)
        ax.set_title("UMAP Projection by Rhetorical Role")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.legend(title="Role", loc="best")
        plt.tight_layout()
        return fig

    def _frequency_bar(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Bar chart of sentence frequency per cluster sorted descending.

        Args:
            statistics: StatisticsOutput with cluster_stats list

        Returns:
            matplotlib Figure
        """
        stats = sorted(statistics.cluster_stats, key=lambda s: s.frequency, reverse=True)
        fig, ax = plt.subplots(figsize=self._figsize)
        if not stats:
            ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Cluster Sentence Frequency")
            plt.tight_layout()
            return fig
        labels = [f"{s.role[:3]}-{s.cluster_id}" for s in stats]
        values = [s.frequency for s in stats]
        colors = [_ROLE_COLORS.get(s.role, "#999999") for s in stats]
        ax.bar(labels, values, color=colors)
        ax.set_title("Cluster Sentence Frequency")
        ax.set_xlabel("Cluster (Role-ID)")
        ax.set_ylabel("Sentence Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    def _role_heatmap(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Heatmap of role distribution across clusters.

        Args:
            statistics: StatisticsOutput with cluster_stats list

        Returns:
            matplotlib Figure
        """
        from step_5_rhetorical_labeler import ROLES

        stats = statistics.cluster_stats
        fig, ax = plt.subplots(figsize=self._figsize)
        if not stats:
            ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Role Distribution Heatmap Across Clusters")
            plt.tight_layout()
            return fig
        cluster_names = [f"{s.role[:3]}-{s.cluster_id}" for s in stats]
        matrix = np.array([
            [s.role_distribution.get(role, 0) for role in ROLES]
            for s in stats
        ], dtype=float).reshape(-1, len(ROLES))
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_matrix = matrix / row_sums
        img = ax.imshow(norm_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(cluster_names)))
        ax.set_xticklabels(cluster_names, rotation=45, ha="right")
        ax.set_yticks(range(len(ROLES)))
        ax.set_yticklabels(ROLES)
        ax.set_title("Role Distribution Heatmap Across Clusters")
        ax.set_xlabel("Cluster (Role-ID)")
        ax.set_ylabel("Rhetorical Role")
        fig.colorbar(img, ax=ax, label="Proportion")
        plt.tight_layout()
        return fig

    def validate(self, output_data: VisualizationOutput) -> bool:
        """
        Validate that all three figures were produced.

        Args:
            output_data: VisualizationOutput to validate

        Returns:
            True if exactly three figures were generated
        """
        return len(output_data.figures) == 3
