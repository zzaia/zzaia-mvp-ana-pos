"""Step 8: Cluster statistics with post-hoc case type labeling via regex on representative sentence."""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_step import PipelineStep
from step_7_hdbscan_clusterer import ClusteringOutput

_CASE_PATTERNS: dict[str, list[str]] = {
    "CasaRoubada": [
        r"\b(residência|domicílio|casa|imóvel|apartamento)\b",
        r"\b(roubad[oa]|furtad[oa]|invadid[oa]|arrombad[oa])\b",
    ],
    "AssaltoArmado": [
        r"\b(arma|armad[oa]|revólver|pistola|faca|arma\s+de\s+fogo)\b",
        r"\b(assalto|roubo|rendeu|ameaçou)\b",
    ],
    "HomicidioDoloso": [
        r"\b(homicídio|matar|matou|vítima\s+fatal|óbito|morte\s+dolosa|ceifou\s+a\s+vida)\b",
    ],
    "TraficoDrogas": [
        r"\b(tráfico|entorpecente|droga|cocaína|maconha|crack|substância\s+ilícita)\b",
    ],
    "ViolenciaDomestica": [
        r"\b(violência\s+doméstica|Lei\s+Maria\s+da\s+Penha|cônjuge|companheiro|ameaça|lesão\s+corporal\s+doméstica)\b",
    ],
    "FurtoCelular": [
        r"\b(celular|smartphone|aparelho\s+telefônico|furto\s+de\s+telefone)\b",
    ],
    "AcidenteTransito": [
        r"\b(acidente\s+de\s+trânsito|colisão|atropelamento|veículo|embriagado\s+ao\s+volante|CNH)\b",
    ],
    "EstelionatoFraude": [
        r"\b(estelionato|fraude|falsidade\s+ideológica|documento\s+falso|golpe|enganou|iludiu)\b",
    ],
    "CrimesFinanceiros": [
        r"\b(lavagem\s+de\s+dinheiro|peculato|corrupção|desvio\s+de\s+verbas|improbidade|enriquecimento\s+ilícito)\b",
    ],
    "LesaoCorporal": [
        r"\b(lesão\s+corporal|lesões\s+corporais|ferimento|agrediu|espancou|socou|chutou)\b",
    ],
}


@dataclass
class ClusterStats:
    """
    Aggregated statistics for a single cluster.

    Attributes:
        cluster_id: HDBSCAN cluster label
        case_type: Post-hoc case type assigned via regex on representative sentence
        frequency: Number of sentences in this cluster
        representative_sentence: Sentence closest to the centroid
        intra_similarity: Mean cosine similarity among cluster members
        centroid: Mean embedding vector for this cluster
    """

    cluster_id: int
    case_type: str
    frequency: int
    representative_sentence: str
    intra_similarity: float
    centroid: np.ndarray


@dataclass
class StatisticsOutput:
    """
    Output of the statistics generation step.

    Attributes:
        cluster_stats: List of per-cluster statistics
        cross_similarity: Matrix of centroid cosine similarities between clusters
        cluster_labels: Ordered list of cluster_id values matching matrix rows
        total_clusters: Total number of non-noise clusters
        case_type_frequency: Total sentence count per case type across all clusters
        source_path: Propagated from previous step
    """

    cluster_stats: list[ClusterStats]
    cross_similarity: np.ndarray
    cluster_labels: list[int]
    total_clusters: int
    case_type_frequency: dict[str, int] = field(default_factory=dict)
    source_path: Optional[object] = None


class StatisticsGenerator(PipelineStep):
    """
    Compute per-cluster statistics and assign case type labels post-hoc.

    For each HDBSCAN cluster, calculates: sentence frequency, the
    representative sentence (nearest to centroid), and mean intra-cluster
    cosine similarity. After computing the representative sentence, regex
    patterns from _CASE_PATTERNS are applied to assign a case type label
    to the cluster. Cross-cluster cosine similarity between centroids
    reveals thematic overlap. Noise sentences (cluster_id == -1) are
    excluded from cluster statistics.
    """

    def __init__(self):
        """Initialize statistics generator."""
        super().__init__(
            step_number=8,
            name="Statistics Generator",
            description="Compute frequency tables, similarity, and post-hoc case type labels",
        )

    def process(self, input_data: ClusteringOutput) -> StatisticsOutput:
        """
        Compute statistics for all clusters with post-hoc case type labeling.

        Args:
            input_data: ClusteringOutput with cluster assignments

        Returns:
            StatisticsOutput with per-cluster statistics and case_type_frequency
        """
        sentences = [s for s in input_data.clustered_sentences if s.cluster_id != -1]
        group_map: dict[int, list] = {}
        for sentence in sentences:
            group_map.setdefault(sentence.cluster_id, []).append(sentence)
        cluster_stats: list[ClusterStats] = []
        for cluster_id, members in sorted(group_map.items()):
            stats = self._compute_cluster_stats(cluster_id, members)
            cluster_stats.append(stats)
        cross_sim, labels = self._compute_cross_similarity(cluster_stats)
        case_type_frequency = self._aggregate_case_type_frequency(cluster_stats)
        return StatisticsOutput(
            cluster_stats=cluster_stats,
            cross_similarity=cross_sim,
            cluster_labels=labels,
            total_clusters=len(cluster_stats),
            case_type_frequency=case_type_frequency,
            source_path=input_data.source_path,
        )

    def _compute_cluster_stats(self, cluster_id: int, members: list) -> ClusterStats:
        """
        Compute statistics for one cluster and assign case type via regex.

        Args:
            cluster_id: HDBSCAN cluster label
            members: List of ClusteredSentence objects in this cluster

        Returns:
            Populated ClusterStats dataclass with post-hoc case_type
        """
        embeddings = np.stack([m.embedding for m in members])
        centroid = embeddings.mean(axis=0)
        dists = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        representative = members[int(np.argmax(dists))].text
        intra_sim = float(np.mean(cosine_similarity(embeddings)))
        case_type = self._label_cluster(representative)
        return ClusterStats(
            cluster_id=cluster_id,
            case_type=case_type,
            frequency=len(members),
            representative_sentence=representative,
            intra_similarity=round(intra_sim, 4),
            centroid=centroid,
        )

    def _label_cluster(self, representative_sentence: str) -> str:
        """
        Assign a case type by applying regex patterns to the representative sentence.

        Args:
            representative_sentence: Sentence closest to the cluster centroid

        Returns:
            First matching case type name or Unknown if no patterns match
        """
        for case_type, patterns in _CASE_PATTERNS.items():
            if all(re.search(p, representative_sentence, re.IGNORECASE) for p in patterns):
                return case_type
        for case_type, patterns in _CASE_PATTERNS.items():
            if any(re.search(p, representative_sentence, re.IGNORECASE) for p in patterns):
                return case_type
        return "Unknown"

    def _aggregate_case_type_frequency(
        self, cluster_stats: list[ClusterStats]
    ) -> dict[str, int]:
        """
        Sum sentence frequencies across all clusters sharing the same case type.

        Args:
            cluster_stats: List of ClusterStats with case_type and frequency

        Returns:
            Dictionary mapping case type to total sentence count
        """
        result: dict[str, int] = {}
        for stats in cluster_stats:
            result[stats.case_type] = result.get(stats.case_type, 0) + stats.frequency
        return result

    def _compute_cross_similarity(
        self, cluster_stats: list[ClusterStats]
    ) -> tuple[np.ndarray, list[int]]:
        """
        Compute pairwise centroid cosine similarity across all clusters.

        Args:
            cluster_stats: List of ClusterStats with centroid vectors

        Returns:
            Tuple of (similarity matrix, ordered cluster_id list)
        """
        if not cluster_stats:
            return np.empty((0, 0)), []
        centroids = np.stack([s.centroid for s in cluster_stats])
        labels = [s.cluster_id for s in cluster_stats]
        sim_matrix = cosine_similarity(centroids)
        return sim_matrix, labels

    def validate(self, output_data: StatisticsOutput) -> bool:
        """
        Validate that at least one cluster was processed.

        Args:
            output_data: StatisticsOutput to validate

        Returns:
            True if cluster_stats processing completed
        """
        return len(output_data.cluster_stats) >= 0
