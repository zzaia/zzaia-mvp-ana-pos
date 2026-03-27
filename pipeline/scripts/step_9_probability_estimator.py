"""Step 9: Probabilistic query occurrence estimator over STJ Súmulas corpus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.stats import gaussian_kde

from step_7_search_index import SearchIndexOutput
from visualization_step import VisualizationStep


@dataclass
class QueryProbability:
    """
    Probabilistic occurrence estimate for a single query.

    Attributes:
        query: The original query string
        softmax_probability: P(query) from softmax of mean similarities across all queries
        kde_probability: P(sim > corpus_mean | query) estimated via Gaussian KDE
        z_score: Standard deviations above the corpus-wide similarity baseline
        combined_score: Weighted combination of the three metrics normalized to [0, 1]
        rank: 1-based rank (1 = most likely to appear in the document)
    """

    query: str
    softmax_probability: float
    kde_probability: float
    z_score: float
    combined_score: float
    rank: int


@dataclass
class ProbabilityEstimatorOutput:
    """
    Output of the probability estimator step.

    Attributes:
        query_probabilities: List of QueryProbability sorted by rank ascending
        corpus_mean: Grand mean of all similarity values across all queries
        corpus_std: Grand standard deviation of all similarity values
        figures: List of matplotlib figures produced
        saved_paths: Paths to saved figure files
    """

    query_probabilities: list[QueryProbability]
    corpus_mean: float
    corpus_std: float
    figures: list[Any]
    saved_paths: list[Path] = field(default_factory=list)

    @property
    def winner(self) -> QueryProbability:
        """The query most likely to appear in the document (rank 1)."""
        return self.query_probabilities[0]


class ProbabilityEstimator(VisualizationStep):
    """
    Estimate the probability that each query topic appears in the súmula corpus.

    Combines three complementary probabilistic metrics:
    1. Softmax probability: P(query) = exp(mean_sim) / sum(exp(mean_sim_j))
    2. KDE probability: area under Gaussian KDE above the corpus-wide mean
    3. Z-score: (mean_sim - corpus_mean) / corpus_std

    A weighted combined score ranks all queries. The top-ranked query is the
    one most likely to appear as a case theme in the STJ Súmulas document.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize probability estimator.

        Args:
            output_dir: Directory for saving figure images; created if absent
        """
        super().__init__(
            step_number=9,
            name="Probability Estimator",
            description="Estimate query occurrence probability via KDE and softmax",
            output_dir=output_dir,
        )

    def _kde_probability_above(
        self, similarities: list[float], baseline: float
    ) -> float:
        """
        Estimate P(similarity > baseline) using scipy Gaussian KDE.

        Args:
            similarities: List of cosine similarity values
            baseline: Lower integration limit (corpus mean)

        Returns:
            Probability mass above baseline under the KDE
        """
        arr = np.array(similarities, dtype=float)
        if arr.std() == 0.0 or len(arr) < 2:
            return 1.0 if float(arr.mean()) > baseline else 0.0
        kde = gaussian_kde(arr)
        x = np.linspace(baseline, float(arr.max()) + 0.05, 500)
        return float(np.trapz(kde(x), x))

    def _min_max_normalize(self, values: list[float]) -> list[float]:
        """
        Min-max normalize a list of values to [0, 1].

        Args:
            values: Input values

        Returns:
            Values scaled to [0, 1]
        """
        arr = np.array(values, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return [0.5] * len(values)
        return list((arr - lo) / (hi - lo))

    def _build_ranking_figure(
        self, probabilities: list[QueryProbability], corpus_mean: float
    ) -> matplotlib.figure.Figure:
        """
        Build a three-panel horizontal bar chart comparing all queries.

        Panels show softmax probability, KDE probability, and Z-score side by side.
        The winning query is highlighted in a different color.

        Args:
            probabilities: List of QueryProbability sorted by rank ascending
            corpus_mean: Grand corpus mean for annotation

        Returns:
            matplotlib Figure with the ranking visualization
        """
        labels = [
            f"Q{p.rank}: {p.query[:55]}..." if len(p.query) > 55 else f"Q{p.rank}: {p.query}"
            for p in probabilities
        ]
        softmax_vals = [p.softmax_probability for p in probabilities]
        kde_vals = [p.kde_probability for p in probabilities]
        z_vals = [p.z_score for p in probabilities]
        colors = ["#2ecc71" if p.rank == 1 else "steelblue" for p in probabilities]
        n = len(probabilities)
        fig, raw_axes = plt.subplots(nrows=1, ncols=3, figsize=(16, max(4, n * 0.9 + 2)))
        axes: list[matplotlib.axes.Axes] = list(raw_axes)
        y_pos = list(range(n - 1, -1, -1))
        for ax, vals, title, fmt in [
            (axes[0], softmax_vals, "Softmax Probability\nP(query) = exp(mean_sim) / Σ exp", ".4f"),
            (axes[1], kde_vals, "KDE Probability\nP(sim > corpus_mean | query)", ".4f"),
            (axes[2], z_vals, "Z-Score\n(mean_sim − corpus_mean) / σ", ".3f"),
        ]:
            bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_width() + max(vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    format(val, fmt),
                    va="center", fontsize=8,
                )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(title, fontsize=9, pad=8)
            ax.set_xlabel("Score", fontsize=8)
        fig.suptitle(
            f'Query Occurrence Probability — Winner: "{probabilities[0].query}"\n'
            f"(corpus mean similarity = {corpus_mean:.4f})",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        return fig

    def process(self, input_data: list[SearchIndexOutput]) -> ProbabilityEstimatorOutput:
        """
        Compute probabilistic occurrence estimates for all queries.

        Args:
            input_data: List of SearchIndexOutput, one per query

        Returns:
            ProbabilityEstimatorOutput with ranked QueryProbability list and figures
        """
        all_sims = [r.similarity for o in input_data for r in o.results]
        corpus_mean = float(np.mean(all_sims))
        corpus_std = float(np.std(all_sims)) or 1e-9
        mean_sims = [o.mean_similarity for o in input_data]
        softmax_probs = list(scipy_softmax(np.array(mean_sims, dtype=float)))
        kde_probs = [
            self._kde_probability_above(
                [r.similarity for r in o.results], corpus_mean
            )
            for o in input_data
        ]
        z_scores = [(m - corpus_mean) / corpus_std for m in mean_sims]
        norm_softmax = self._min_max_normalize(softmax_probs)
        norm_kde = self._min_max_normalize(kde_probs)
        norm_z = self._min_max_normalize(z_scores)
        combined = [0.4 * s + 0.4 * k + 0.2 * z for s, k, z in zip(norm_softmax, norm_kde, norm_z)]
        ranked_indices = sorted(range(len(input_data)), key=lambda i: combined[i], reverse=True)
        query_probs: list[QueryProbability] = []
        rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked_indices)}
        for i, output in enumerate(input_data):
            query_probs.append(QueryProbability(
                query=output.query,
                softmax_probability=softmax_probs[i],
                kde_probability=kde_probs[i],
                z_score=z_scores[i],
                combined_score=combined[i],
                rank=rank_map[i],
            ))
        sorted_probs = sorted(query_probs, key=lambda p: p.rank)
        figure = self._build_ranking_figure(sorted_probs, corpus_mean)
        saved_paths: list[Path] = []
        path = self._save_figure(figure, "query_probability_ranking.png")
        if path:
            saved_paths.append(path)
        return ProbabilityEstimatorOutput(
            query_probabilities=sorted_probs,
            corpus_mean=corpus_mean,
            corpus_std=corpus_std,
            figures=[figure],
            saved_paths=saved_paths,
        )

    def validate(self, output_data: ProbabilityEstimatorOutput) -> bool:
        """
        Validate that at least one query was ranked.

        Args:
            output_data: ProbabilityEstimatorOutput to validate

        Returns:
            True if query_probabilities is non-empty
        """
        return len(output_data.query_probabilities) > 0
