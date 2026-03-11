"""Step 8: HDBSCAN clustering applied per rhetorical role group."""

from dataclasses import dataclass, field
from typing import Optional

import hdbscan
import numpy as np

from pipeline_step import PipelineStep
from step_7_umap_reducer import ReducedSentence, UmapOutput


@dataclass
class ClusteredSentence:
    """
    A sentence annotated with its HDBSCAN cluster assignment.

    Attributes:
        text: Sentence text
        role: Rhetorical role label
        confidence: Role classification confidence
        embedding: Original 768-dim embedding
        reduced_vector: UMAP-reduced vector
        cluster_id: Cluster label (-1 means noise)
        cluster_probability: HDBSCAN soft cluster membership probability
        citation_metadata: Original citations
    """

    text: str
    role: str
    confidence: float
    embedding: np.ndarray
    reduced_vector: np.ndarray
    cluster_id: int
    cluster_probability: float
    citation_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ClusteringOutput:
    """
    Output of the HDBSCAN clustering step.

    Attributes:
        clustered_sentences: Sentences with cluster assignments
        cluster_counts: Per-role cluster count statistics
        noise_count: Total sentences labeled as noise (cluster_id == -1)
        min_cluster_size: HDBSCAN parameter used
        min_samples: HDBSCAN parameter used
        source_path: Propagated from previous step
    """

    clustered_sentences: list[ClusteredSentence]
    cluster_counts: dict[str, int] = field(default_factory=dict)
    noise_count: int = 0
    min_cluster_size: int = 5
    min_samples: int = 3
    source_path: Optional[object] = None


class HdbscanClusterer(PipelineStep):
    """
    Cluster sentences within each rhetorical role using HDBSCAN.

    HDBSCAN is applied independently per role group on the UMAP-reduced
    vectors. This avoids forcing all roles into a single global topology
    and lets cluster density thresholds be role-specific. Euclidean
    distance is used on the low-dimensional UMAP output (cosine similarity
    was applied during UMAP reduction). Sentences not belonging to any
    cluster are assigned cluster_id = -1.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        """
        Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum points to form a cluster
            min_samples: Minimum samples in a neighbourhood
        """
        super().__init__(
            step_number=8,
            name="HDBSCAN Clusterer",
            description="Cluster sentences per role group with HDBSCAN",
        )
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples

    def process(self, input_data: UmapOutput) -> ClusteringOutput:
        """
        Assign clusters to all sentences grouped by rhetorical role.

        Args:
            input_data: UmapOutput with UMAP-reduced vectors

        Returns:
            ClusteringOutput with cluster assignments per sentence
        """
        sentences = input_data.reduced_sentences
        role_groups: dict[str, list[int]] = {}
        for idx, item in enumerate(sentences):
            role_groups.setdefault(item.role, []).append(idx)
        label_map: dict[int, int] = {}
        prob_map: dict[int, float] = {}
        cluster_counts: dict[str, int] = {}
        for role, indices in role_groups.items():
            matrix = np.stack([sentences[i].reduced_vector for i in indices])
            labels, probs = self._cluster_group(matrix)
            cluster_counts[role] = int(np.max(labels) + 1) if np.max(labels) >= 0 else 0
            for local_idx, global_idx in enumerate(indices):
                label_map[global_idx] = int(labels[local_idx])
                prob_map[global_idx] = float(probs[local_idx])
        clustered = [
            ClusteredSentence(
                text=item.text,
                role=item.role,
                confidence=item.confidence,
                embedding=item.embedding,
                reduced_vector=item.reduced_vector,
                cluster_id=label_map[idx],
                cluster_probability=prob_map[idx],
                citation_metadata=item.citation_metadata,
            )
            for idx, item in enumerate(sentences)
        ]
        noise_count = sum(1 for s in clustered if s.cluster_id == -1)
        return ClusteringOutput(
            clustered_sentences=clustered,
            cluster_counts=cluster_counts,
            noise_count=noise_count,
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            source_path=input_data.source_path,
        )

    def _cluster_group(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit HDBSCAN on a role group matrix.

        Args:
            matrix: (n_samples, n_components) reduced embedding matrix

        Returns:
            Tuple of (cluster_labels, membership_probabilities)
        """
        if len(matrix) < self._min_cluster_size:
            return np.full(len(matrix), -1), np.zeros(len(matrix))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric="euclidean",
            prediction_data=True,
        )
        clusterer.fit(matrix)
        return clusterer.labels_, clusterer.probabilities_

    def validate(self, output_data: ClusteringOutput) -> bool:
        """
        Validate that clustering produced output for all sentences.

        Args:
            output_data: ClusteringOutput to validate

        Returns:
            True if clustered_sentences is non-empty
        """
        return len(output_data.clustered_sentences) > 0
