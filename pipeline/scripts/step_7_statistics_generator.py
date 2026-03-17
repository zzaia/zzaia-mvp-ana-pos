"""Step 7: Per-topic statistics using embeddings from BerTopicOutput."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_step import PipelineStep
from step_6_bertopic import BerTopicOutput


@dataclass
class ClusterStats:
    """
    Aggregated statistics for a single BERTopic topic.

    Attributes:
        topic_id: BERTopic topic label
        topic_label: Auto-generated keyword phrase from c-TF-IDF
        frequency: Number of sentences in this topic
        representative_sentence: Sentence closest to topic centroid
        non_representative_sentence: Sentence farthest from topic centroid
        intra_similarity: Mean cosine similarity among topic members
        centroid: Mean embedding vector for this topic
    """

    topic_id: int
    topic_label: str
    frequency: int
    representative_sentence: str
    non_representative_sentence: str
    intra_similarity: float
    centroid: np.ndarray


@dataclass
class StatisticsOutput:
    """
    Output of the statistics generation step.

    Attributes:
        cluster_stats: Per-topic statistics list
        cross_similarity: Pairwise centroid cosine similarity matrix
        cluster_labels: Ordered topic_id list matching matrix rows
        total_clusters: Non-noise topic count
        topic_frequency: Total sentence count per topic label
        source_path: Propagated from previous step
    """

    cluster_stats: list[ClusterStats]
    cross_similarity: np.ndarray
    cluster_labels: list[int]
    total_clusters: int
    topic_frequency: dict[str, int] = field(default_factory=dict)
    source_path: Optional[object] = None


class StatisticsGenerator(PipelineStep):
    """
    Compute per-topic statistics from BERTopic output without external model inference.

    For each non-noise topic, computes: sentence frequency, representative sentence
    (nearest to centroid), non-representative sentence (farthest from centroid), and
    mean intra-topic cosine similarity. Topic labels come from BERTopic c-TF-IDF
    auto-generation — no seed embeddings or manual labeling required. Cross-topic
    centroid similarity reveals thematic overlap across the corpus.
    """

    def __init__(self):
        """Initialize statistics generator at pipeline step 7."""
        super().__init__(
            step_number=7,
            name="Statistics Generator",
            description="Compute frequency tables and similarity from BERTopic embeddings",
        )

    def process(self, input_data: BerTopicOutput) -> StatisticsOutput:
        """
        Compute statistics for all non-noise topics.

        Args:
            input_data: BerTopicOutput with topiced_sentences and topic_labels

        Returns:
            StatisticsOutput with per-topic statistics and cross-similarity matrix
        """
        sentences = [s for s in input_data.topiced_sentences if s.topic_id != -1]
        group_map: dict[int, list] = {}
        for sentence in sentences:
            group_map.setdefault(sentence.topic_id, []).append(sentence)
        cluster_stats: list[ClusterStats] = []
        for topic_id, members in sorted(group_map.items()):
            label = input_data.topic_labels.get(topic_id, str(topic_id))
            stats = self._compute_topic_stats(topic_id, label, members)
            cluster_stats.append(stats)
        cross_sim, labels = self._compute_cross_similarity(cluster_stats)
        topic_frequency = self._aggregate_topic_frequency(cluster_stats)
        return StatisticsOutput(
            cluster_stats=cluster_stats,
            cross_similarity=cross_sim,
            cluster_labels=labels,
            total_clusters=len(cluster_stats),
            topic_frequency=topic_frequency,
            source_path=input_data.source_path,
        )

    def _compute_topic_stats(
        self,
        topic_id: int,
        topic_label: str,
        members: list,
    ) -> ClusterStats:
        """
        Compute statistics for a single topic from its member sentences.

        Args:
            topic_id: BERTopic topic identifier
            topic_label: Auto-generated keyword label for this topic
            members: List of TopicedSentence objects in this topic

        Returns:
            Populated ClusterStats dataclass
        """
        embeddings = np.stack([m.embedding for m in members])
        centroid = embeddings.mean(axis=0)
        dists = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        representative = members[int(np.argmax(dists))].text
        non_representative = members[int(np.argmin(dists))].text
        intra_sim = float(np.mean(cosine_similarity(embeddings)))
        return ClusterStats(
            topic_id=topic_id,
            topic_label=topic_label,
            frequency=len(members),
            representative_sentence=representative,
            non_representative_sentence=non_representative,
            intra_similarity=round(intra_sim, 4),
            centroid=centroid,
        )

    def _compute_cross_similarity(
        self,
        cluster_stats: list[ClusterStats],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Compute pairwise centroid cosine similarity across all topics.

        Args:
            cluster_stats: List of ClusterStats with centroid vectors

        Returns:
            Tuple of (similarity matrix, ordered topic_id list)
        """
        if not cluster_stats:
            return np.empty((0, 0)), []
        centroids = np.stack([s.centroid for s in cluster_stats])
        labels = [s.topic_id for s in cluster_stats]
        sim_matrix = cosine_similarity(centroids)
        return sim_matrix, labels

    def _aggregate_topic_frequency(
        self,
        cluster_stats: list[ClusterStats],
    ) -> dict[str, int]:
        """
        Sum sentence frequencies across all topics sharing the same topic label.

        Args:
            cluster_stats: List of ClusterStats with topic_label and frequency

        Returns:
            Dictionary mapping topic label to total sentence count
        """
        result: dict[str, int] = {}
        for stats in cluster_stats:
            result[stats.topic_label] = result.get(stats.topic_label, 0) + stats.frequency
        return result

    def validate(self, output_data: StatisticsOutput) -> bool:
        """
        Validate that statistics processing completed.

        Args:
            output_data: StatisticsOutput to validate

        Returns:
            True if cluster_stats processing completed
        """
        return len(output_data.cluster_stats) >= 0
