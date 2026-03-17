"""Step 8: Visualization of BERTopic scatter, frequency bar chart, and topic frequency chart."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt

from pipeline_step import PipelineStep
from step_6_bertopic import BerTopicOutput
from step_7_statistics_generator import StatisticsOutput


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
    Generate three key visualizations for the BERTopic semantic frequency analysis.

    Produces:
    1. UMAP scatter plot coloured by BERTopic topic using tab20 colormap
    2. Frequency bar chart of sentence counts per topic sorted descending
    3. Topic label frequency bar chart from StatisticsOutput.topic_frequency

    The BERTopic output is injected at construction time since it originates
    from step 6, two steps before this step runs in the pipeline.
    """

    def __init__(
        self,
        bertopic_output: BerTopicOutput,
        output_dir: Optional[Path] = None,
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        Initialize visualizer with injected BERTopic dependency.

        Args:
            bertopic_output: BerTopicOutput from step 6 with UMAP coordinates and topic labels
            output_dir: Directory where figures will be saved (optional)
            figsize: Default matplotlib figure size (width, height)
        """
        super().__init__(
            step_number=8,
            name="Visualizer",
            description="Generate UMAP scatter, frequency chart, and topic frequency chart",
        )
        self._bertopic_output = bertopic_output
        self._output_dir = output_dir
        self._figsize = figsize

    def process(self, input_data: StatisticsOutput) -> VisualizationOutput:
        """
        Produce all three figures using injected BERTopic output.

        Args:
            input_data: StatisticsOutput from step 7 with topic statistics

        Returns:
            VisualizationOutput with figures and optional saved paths
        """
        figures: list[matplotlib.figure.Figure] = []
        saved: list[Path] = []
        figures.append(self._umap_scatter(input_data))
        figures.append(self._frequency_bar(input_data))
        figures.append(self._case_type_frequency_bar(input_data))
        if self._output_dir is not None:
            output_dir = Path(self._output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            names = ["umap_scatter.png", "frequency_bar.png", "topic_frequency.png"]
            for fig, name in zip(figures, names):
                path = output_dir / name
                fig.savefig(path, dpi=150, bbox_inches="tight")
                saved.append(path)
        return VisualizationOutput(figures=figures, saved_paths=saved)

    def _umap_scatter(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Plot 2D UMAP projection coloured by BERTopic topic using tab20 colormap.

        Args:
            statistics: StatisticsOutput with topic statistics for context

        Returns:
            matplotlib Figure
        """
        sentences = self._bertopic_output.topiced_sentences
        fig, ax = plt.subplots(figsize=self._figsize)
        if not sentences:
            ax.text(0.5, 0.5, "No sentences to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("UMAP Projection by BERTopic Topic")
            plt.tight_layout()
            return fig
        unique_topic_ids = sorted({s.topic_id for s in sentences if s.topic_id != -1})
        colors = plt.cm.tab20.colors
        color_map: dict[int, tuple] = {
            topic_id: colors[idx % len(colors)]
            for idx, topic_id in enumerate(unique_topic_ids)
        }
        topic_groups: dict[int, list] = {}
        noise_sentences: list = []
        for s in sentences:
            if s.topic_id == -1:
                noise_sentences.append(s)
            else:
                topic_groups.setdefault(s.topic_id, []).append(s)
        if noise_sentences:
            xs = [s.reduced_vector[0] for s in noise_sentences]
            ys = [s.reduced_vector[1] for s in noise_sentences]
            ax.scatter(xs, ys, label="noise", color="#CCCCCC", alpha=0.4, s=15)
        for topic_id, subset in sorted(topic_groups.items()):
            label = self._bertopic_output.topic_labels.get(topic_id, str(topic_id))
            xs = [s.reduced_vector[0] for s in subset]
            ys = [s.reduced_vector[1] for s in subset]
            ax.scatter(xs, ys, label=f"{topic_id}:{label[:12]}", color=color_map[topic_id], alpha=0.6, s=20)
        ax.set_title("UMAP Projection by BERTopic Topic")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.legend(title="Topic", loc="best", fontsize=7)
        plt.tight_layout()
        return fig

    def _frequency_bar(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Bar chart of sentence frequency per topic sorted descending.

        Args:
            statistics: StatisticsOutput with cluster_stats list

        Returns:
            matplotlib Figure
        """
        stats = sorted(statistics.cluster_stats, key=lambda s: s.frequency, reverse=True)
        fig, ax = plt.subplots(figsize=self._figsize)
        if not stats:
            ax.text(0.5, 0.5, "No topics to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Topic Sentence Frequency")
            plt.tight_layout()
            return fig
        labels = [f"{s.topic_label[:8]}-{s.topic_id}" for s in stats]
        values = [s.frequency for s in stats]
        colors = list(plt.cm.tab20.colors)
        bar_colors = [colors[idx % len(colors)] for idx in range(len(stats))]
        ax.bar(labels, values, color=bar_colors)
        ax.set_title("Topic Sentence Frequency")
        ax.set_xlabel("Topic (Label-ID)")
        ax.set_ylabel("Sentence Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    def _case_type_frequency_bar(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Bar chart of total sentence frequency per topic label from topic_frequency.

        Args:
            statistics: StatisticsOutput with topic_frequency dict

        Returns:
            matplotlib Figure
        """
        freq = statistics.topic_frequency
        fig, ax = plt.subplots(figsize=self._figsize)
        if not freq:
            ax.text(0.5, 0.5, "No topic frequency data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Topic Label Frequency")
            plt.tight_layout()
            return fig
        sorted_items = sorted(
            ((k, v) for k, v in freq.items() if k != "-1"),
            key=lambda x: x[1],
            reverse=True,
        )
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = list(plt.cm.tab20.colors)
        bar_colors = [colors[idx % len(colors)] for idx in range(len(labels))]
        ax.bar(labels, values, color=bar_colors)
        ax.set_title("Topic Label Frequency")
        ax.set_xlabel("Topic Label")
        ax.set_ylabel("Total Sentence Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    def validate(self, output_data: VisualizationOutput) -> bool:
        """
        Validate that all three figures were produced.

        Args:
            output_data: VisualizationOutput to validate

        Returns:
            True if figures list is non-empty
        """
        return len(output_data.figures) > 0
