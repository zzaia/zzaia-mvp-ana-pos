"""Step 7: HDBSCAN clustering applied to all sentences in a single run."""

from dataclasses import dataclass, field
from typing import Optional

import hdbscan
import numpy as np

from pipeline_step import PipelineStep
from step_6_umap_reducer import ReducedSentence, UmapOutput


@dataclass
class ClusteredSentence:
    """
    A sentence annotated with its HDBSCAN cluster assignment.

    Attributes:
        text: Sentence text
        embedding: Original 768-dim embedding
        reduced_vector: UMAP-reduced vector
        cluster_id: Cluster label (-1 means noise)
        membership_probability: HDBSCAN soft cluster membership probability
        citation_metadata: Original citations
    """

    text: str
    embedding: np.ndarray
    reduced_vector: np.ndarray
    cluster_id: int
    membership_probability: float
    citation_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ClusteringOutput:
    """
    Output of the HDBSCAN clustering step.

    Attributes:
        clustered_sentences: Sentences with cluster assignments
        total_clusters: Total number of non-noise clusters found
        noise_count: Total sentences labeled as noise (cluster_id == -1)
        min_cluster_size: HDBSCAN parameter used
        min_samples: HDBSCAN parameter used
        source_path: Propagated from previous step
    """

    clustered_sentences: list[ClusteredSentence]
    total_clusters: int = 0
    noise_count: int = 0
    min_cluster_size: int = 5
    min_samples: int = 3
    source_path: Optional[object] = None


class HdbscanClusterer(PipelineStep):
    """
    Cluster all sentences in a single HDBSCAN run.

    HDBSCAN is applied to the full UMAP-reduced corpus in one pass, allowing
    the algorithm to discover density-based clusters across the entire dataset
    without prior case type knowledge. Euclidean distance is used on the
    low-dimensional UMAP output. Sentences not belonging to any cluster are
    assigned cluster_id = -1. Post-hoc case type labeling is performed in
    Step 8 using the representative sentence of each cluster.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        """
        Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum points to form a cluster
            min_samples: Minimum samples in a neighbourhood
        """
        super().__init__(
            step_number=7,
            name="HDBSCAN Clusterer",
            description="Cluster all sentences in a single HDBSCAN run",
        )
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples

    def process(self, input_data: UmapOutput) -> ClusteringOutput:
        """
        Assign clusters to all sentences in a single HDBSCAN run.

        Args:
            input_data: UmapOutput with UMAP-reduced vectors

        Returns:
            ClusteringOutput with cluster assignments per sentence
        """
        sentences = input_data.reduced_sentences
        matrix = np.stack([item.reduced_vector for item in sentences])
        labels, probs = self._cluster(matrix)
        clustered = [
            ClusteredSentence(
                text=item.text,
                embedding=item.embedding,
                reduced_vector=item.reduced_vector,
                cluster_id=int(labels[idx]),
                membership_probability=float(probs[idx]),
                citation_metadata=item.citation_metadata,
            )
            for idx, item in enumerate(sentences)
        ]
        noise_count = sum(1 for s in clustered if s.cluster_id == -1)
        total_clusters = int(np.max(labels) + 1) if np.max(labels) >= 0 else 0
        return ClusteringOutput(
            clustered_sentences=clustered,
            total_clusters=total_clusters,
            noise_count=noise_count,
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            source_path=input_data.source_path,
        )

    def _cluster(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit HDBSCAN on the full reduced embedding matrix.

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
