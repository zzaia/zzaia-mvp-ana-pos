"""Step 9: Visualization of UMAP scatter, frequency bar chart, and case type heatmap."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.figure

from pipeline_step import PipelineStep
from step_7_hdbscan_clusterer import ClusteringOutput
from step_8_statistics_generator import StatisticsOutput

_CASE_COLORS = {
    "CasaRoubada": "#4C72B0",
    "AssaltoArmado": "#DD8452",
    "HomicidioDoloso": "#55A868",
    "TraficoDrogas": "#C44E52",
    "ViolenciaDomestica": "#8172B3",
    "FurtoCelular": "#937860",
    "AcidenteTransito": "#DA8BC3",
    "EstelionatoFraude": "#8C8C8C",
    "CrimesFinanceiros": "#CCB974",
    "LesaoCorporal": "#64B5CD",
    "DireitoAdministrativo": "#2166AC",
    "DireitoTributario": "#4DAC26",
    "DireitoPrevidenciario": "#D01C8B",
    "DireitoConsumidor": "#F1B6DA",
    "DireitoCivil": "#B8E186",
    "DireitoProcessual": "#762A83",
    "DireitoContratual": "#E66101",
    "DireitoAmbiental": "#1A9641",
    "DireitoTrabalhista": "#D7191C",
    "DireitoConstitucional": "#2C7BB6",
    "Unknown": "#CCCCCC",
}


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
    1. UMAP scatter plot coloured by post-hoc case type from ClusterStats
    2. Frequency bar chart of sentence counts per cluster
    3. Case type frequency bar chart from StatisticsOutput.case_type_frequency

    The clustering output is injected at construction time since it originates
    from step 7, two steps before this step runs in the pipeline.
    """

    def __init__(
        self,
        clustering_output: ClusteringOutput,
        output_dir: Optional[Path] = None,
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        Initialize visualizer with injected clustering dependency.

        Args:
            clustering_output: ClusteringOutput from step 7 with UMAP coordinates
            output_dir: Directory where figures will be saved (optional)
            figsize: Default matplotlib figure size (width, height)
        """
        super().__init__(
            step_number=9,
            name="Visualizer",
            description="Generate UMAP scatter, frequency chart, and case type heatmap",
        )
        self._clustering_output = clustering_output
        self._output_dir = output_dir
        self._figsize = figsize

    def process(self, input_data: StatisticsOutput) -> VisualizationOutput:
        """
        Produce all three figures using injected clustering output.

        Args:
            input_data: StatisticsOutput from step 8 with cluster statistics

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
            names = ["umap_scatter.png", "frequency_bar.png", "case_type_frequency.png"]
            for fig, name in zip(figures, names):
                path = output_dir / name
                fig.savefig(path, dpi=150, bbox_inches="tight")
                saved.append(path)
        return VisualizationOutput(figures=figures, saved_paths=saved)

    def _umap_scatter(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Plot 2D UMAP projection coloured by post-hoc case type from ClusterStats.

        Args:
            statistics: StatisticsOutput with case_type per cluster

        Returns:
            matplotlib Figure
        """
        cluster_case_type: dict[int, str] = {
            s.cluster_id: s.case_type for s in statistics.cluster_stats
        }
        sentences = self._clustering_output.clustered_sentences
        fig, ax = plt.subplots(figsize=self._figsize)
        if not sentences:
            ax.text(0.5, 0.5, "No sentences to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("UMAP Projection by Case Type")
            plt.tight_layout()
            return fig
        case_type_groups: dict[str, list] = {}
        for s in sentences:
            label = cluster_case_type.get(s.cluster_id, "Unknown")
            if label == "Unknown":
                continue
            case_type_groups.setdefault(label, []).append(s)
        for case_type, subset in sorted(case_type_groups.items()):
            xs = [s.reduced_vector[0] for s in subset]
            ys = [s.reduced_vector[1] for s in subset]
            color = _CASE_COLORS.get(case_type, "#999999")
            ax.scatter(xs, ys, label=case_type, color=color, alpha=0.6, s=20)
        ax.set_title("UMAP Projection by Case Type")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.legend(title="Case Type", loc="best")
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
        stats = sorted(
            [s for s in statistics.cluster_stats if s.case_type != "Unknown"],
            key=lambda s: s.frequency,
            reverse=True,
        )
        fig, ax = plt.subplots(figsize=self._figsize)
        if not stats:
            ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Cluster Sentence Frequency")
            plt.tight_layout()
            return fig
        labels = [f"{s.case_type[:4]}-{s.cluster_id}" for s in stats]
        values = [s.frequency for s in stats]
        colors = [_CASE_COLORS.get(s.case_type, "#999999") for s in stats]
        ax.bar(labels, values, color=colors)
        ax.set_title("Cluster Sentence Frequency")
        ax.set_xlabel("Cluster (CaseType-ID)")
        ax.set_ylabel("Sentence Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        return fig

    def _case_type_frequency_bar(self, statistics: StatisticsOutput) -> matplotlib.figure.Figure:
        """
        Bar chart of total sentence frequency per case type from case_type_frequency.

        Args:
            statistics: StatisticsOutput with case_type_frequency dict

        Returns:
            matplotlib Figure
        """
        freq = statistics.case_type_frequency
        fig, ax = plt.subplots(figsize=self._figsize)
        if not freq:
            ax.text(0.5, 0.5, "No case types to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Case Type Frequency (Post-hoc)")
            plt.tight_layout()
            return fig
        sorted_items = sorted(
            ((k, v) for k, v in freq.items() if k != "Unknown"),
            key=lambda x: x[1],
            reverse=True,
        )
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = [_CASE_COLORS.get(label, "#999999") for label in labels]
        ax.bar(labels, values, color=colors)
        ax.set_title("Case Type Frequency (Post-hoc)")
        ax.set_xlabel("Case Type")
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
