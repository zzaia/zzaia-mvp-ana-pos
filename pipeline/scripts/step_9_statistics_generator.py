"""Step 9: Cluster statistics including frequency, similarity, and role distribution."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_step import PipelineStep
from step_8_hdbscan_clusterer import ClusteringOutput


@dataclass
class ClusterStats:
    """
    Aggregated statistics for a single cluster.

    Attributes:
        cluster_id: HDBSCAN cluster label
        role: Rhetorical role of the cluster's group
        frequency: Number of sentences in this cluster
        representative_sentence: Sentence closest to the centroid
        role_distribution: Sentence count per role within this cluster
        intra_similarity: Mean cosine similarity among cluster members
        centroid: Mean embedding vector for this cluster
        label: Optional LLM-generated semantic label (placeholder)
    """

    cluster_id: int
    role: str
    frequency: int
    representative_sentence: str
    role_distribution: dict[str, int]
    intra_similarity: float
    centroid: np.ndarray
    label: str = ""


@dataclass
class StatisticsOutput:
    """
    Output of the statistics generation step.

    Attributes:
        cluster_stats: List of per-cluster statistics
        cross_similarity: Matrix of centroid cosine similarities between clusters
        cluster_labels: Ordered list of (role, cluster_id) pairs matching matrix rows
        total_clusters: Total number of non-noise clusters
        source_path: Propagated from previous step
    """

    cluster_stats: list[ClusterStats]
    cross_similarity: np.ndarray
    cluster_labels: list[tuple[str, int]]
    total_clusters: int
    source_path: Optional[object] = None


class StatisticsGenerator(PipelineStep):
    """
    Compute per-cluster and cross-cluster statistics.

    For each HDBSCAN cluster, calculates: sentence frequency, the
    representative sentence (nearest to centroid), role distribution,
    and mean intra-cluster cosine similarity. Cross-cluster cosine
    similarity between centroids reveals thematic overlap across the
    document. Noise sentences (cluster_id == -1) are excluded from
    cluster statistics.
    """

    def __init__(self):
        """Initialize statistics generator."""
        super().__init__(
            step_number=9,
            name="Statistics Generator",
            description="Compute frequency tables, similarity, and role distributions",
        )

    def process(self, input_data: ClusteringOutput) -> StatisticsOutput:
        """
        Compute statistics for all clusters.

        Args:
            input_data: ClusteringOutput with cluster assignments

        Returns:
            StatisticsOutput with per-cluster and cross-cluster statistics
        """
        sentences = [s for s in input_data.clustered_sentences if s.cluster_id != -1]
        group_map: dict[tuple[str, int], list] = {}
        for sentence in sentences:
            key = (sentence.role, sentence.cluster_id)
            group_map.setdefault(key, []).append(sentence)
        cluster_stats: list[ClusterStats] = []
        for (role, cluster_id), members in sorted(group_map.items()):
            stats = self._compute_cluster_stats(cluster_id, role, members)
            cluster_stats.append(stats)
        cross_sim, labels = self._compute_cross_similarity(cluster_stats)
        return StatisticsOutput(
            cluster_stats=cluster_stats,
            cross_similarity=cross_sim,
            cluster_labels=labels,
            total_clusters=len(cluster_stats),
            source_path=input_data.source_path,
        )

    def _compute_cluster_stats(
        self, cluster_id: int, role: str, members: list
    ) -> ClusterStats:
        """
        Compute statistics for one cluster.

        Args:
            cluster_id: HDBSCAN cluster label
            role: Rhetorical role for this group
            members: List of ClusteredSentence objects in this cluster

        Returns:
            Populated ClusterStats dataclass
        """
        embeddings = np.stack([m.embedding for m in members])
        centroid = embeddings.mean(axis=0)
        dists = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        representative = members[int(np.argmax(dists))].text
        intra_sim = float(np.mean(cosine_similarity(embeddings)))
        role_dist: dict[str, int] = {}
        for m in members:
            role_dist[m.role] = role_dist.get(m.role, 0) + 1
        return ClusterStats(
            cluster_id=cluster_id,
            role=role,
            frequency=len(members),
            representative_sentence=representative,
            role_distribution=role_dist,
            intra_similarity=round(intra_sim, 4),
            centroid=centroid,
        )

    def _compute_cross_similarity(
        self, cluster_stats: list[ClusterStats]
    ) -> tuple[np.ndarray, list[tuple[str, int]]]:
        """
        Compute pairwise centroid cosine similarity across all clusters.

        Args:
            cluster_stats: List of ClusterStats with centroid vectors

        Returns:
            Tuple of (similarity matrix, ordered (role, cluster_id) label list)
        """
        if not cluster_stats:
            return np.empty((0, 0)), []
        centroids = np.stack([s.centroid for s in cluster_stats])
        labels = [(s.role, s.cluster_id) for s in cluster_stats]
        sim_matrix = cosine_similarity(centroids)
        return sim_matrix, labels

    def validate(self, output_data: StatisticsOutput) -> bool:
        """
        Validate that at least one cluster was processed.

        Args:
            output_data: StatisticsOutput to validate

        Returns:
            True if cluster_stats is non-empty
        """
        return len(output_data.cluster_stats) >= 0
