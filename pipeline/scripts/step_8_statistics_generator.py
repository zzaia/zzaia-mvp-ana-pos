"""Step 8: Cluster statistics with post-hoc case type labeling via BERTimbau seed embedding cosine similarity."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from pipeline_step import PipelineStep
from step_7_hdbscan_clusterer import ClusteringOutput

_CASE_SEEDS: dict[str, list[str]] = {
    "CasaRoubada": [
        "O réu arrombou a porta da residência da vítima e subtraiu bens do interior do imóvel.",
        "Os acusados invadiram o domicílio enquanto a família dormia e levaram objetos de valor.",
        "A residência foi furtada durante a madrugada, com arrombamento da janela dos fundos.",
    ],
    "AssaltoArmado": [
        "O acusado, portando revólver calibre 38, rendeu a vítima e subtraiu sua carteira e celular.",
        "Mediante grave ameaça com arma de fogo, os réus praticaram roubo na via pública.",
        "O assaltante armado com faca abordou a vítima e exigiu a entrega de seus pertences.",
    ],
    "HomicidioDoloso": [
        "O réu, com dolo de matar, desferiu facadas na vítima que veio a óbito no local.",
        "O acusado efetuou disparos de arma de fogo com intenção homicida, causando a morte da vítima.",
        "A vítima foi assassinada com múltiplos golpes de instrumento contundente pelo réu.",
    ],
    "TraficoDrogas": [
        "O réu foi preso em flagrante transportando quilogramas de cocaína embalados para venda.",
        "Os acusados mantinham ponto de venda de crack e maconha com divisão de funções entre si.",
        "Foram apreendidos entorpecentes e materiais para embalo, caracterizando tráfico de drogas.",
    ],
    "ViolenciaDomestica": [
        "O réu agrediu sua companheira no ambiente doméstico, causando lesões corporais, respondendo nos termos da Lei Maria da Penha.",
        "O acusado ameaçou de morte sua esposa reiteradamente no contexto de violência doméstica.",
        "A vítima sofreu agressões físicas e psicológicas praticadas pelo cônjuge no lar conjugal.",
    ],
    "FurtoCelular": [
        "O réu subtraiu o aparelho celular da vítima mediante subtração simples em local público.",
        "O acusado furtou smartphone da vítima que estava distraída em estabelecimento comercial.",
        "O celular foi furtado da bolsa da ofendida no interior do transporte público coletivo.",
    ],
    "AcidenteTransito": [
        "O réu conduzia veículo automotor em estado de embriaguez e colidiu com outro automóvel.",
        "O acusado atropelou a vítima na faixa de pedestres ao avançar o sinal vermelho em alta velocidade.",
        "A colisão entre os veículos ocorreu por imprudência do réu que não respeitou a preferencial.",
    ],
    "EstelionatoFraude": [
        "O réu induziu a vítima em erro mediante documentos falsos para obter vantagem patrimonial ilícita.",
        "O acusado aplicou golpe do falso investimento, lesando dezenas de vítimas com promessas fraudulentas.",
        "Mediante falsidade ideológica, o réu obteve crédito bancário em nome de terceiro prejudicado.",
    ],
    "CrimesFinanceiros": [
        "O réu desviou verbas públicas em proveito próprio, configurando peculato doloso.",
        "Os acusados praticaram lavagem de dinheiro ocultando a origem de recursos provenientes de corrupção.",
        "O agente público corrompeu licitações municipais, causando enriquecimento ilícito e dano ao erário.",
    ],
    "LesaoCorporal": [
        "O réu agrediu a vítima com socos e chutes, causando lesões corporais de natureza leve.",
        "O acusado espancou o ofendido em via pública, produzindo ferimentos que necessitaram de atendimento médico.",
        "A vítima sofreu lesões corporais graves em razão de agressão praticada pelo réu com instrumento contundente.",
    ],
    "DireitoAdministrativo": [
        "A Administração Pública deve observar os princípios da legalidade, impessoalidade, moralidade, publicidade e eficiência.",
        "O servidor público tem direito à revisão do ato administrativo que lhe cause prejuízo, assegurado o contraditório.",
        "A anulação de ato administrativo irregular impõe a recomposição do status quo ante para o administrado de boa-fé.",
    ],
    "DireitoTributario": [
        "A taxa de juros de mora incidente na repetição de indébito tributário é a SELIC.",
        "É legítima a cobrança de ICMS sobre a importação de bem destinado ao ativo fixo de empresa não contribuinte.",
        "O Código de Defesa do Consumidor é aplicável às instituições financeiras no tocante a tarifas e encargos.",
    ],
    "DireitoPrevidenciario": [
        "O tempo de serviço rural anterior à Lei de Benefícios pode ser computado para fins de aposentadoria.",
        "É devida a pensão por morte ao cônjuge sobrevivente independentemente do período de carência.",
        "O benefício de prestação continuada é devido ao idoso com renda per capita inferior a um quarto do salário mínimo.",
    ],
    "DireitoConsumidor": [
        "O Código de Defesa do Consumidor se aplica às empresas de planos de saúde.",
        "A inversão do ônus da prova em favor do consumidor é aplicável quando houver verossimilhança da alegação.",
        "É abusiva a cláusula contratual que restringe direitos fundamentais do consumidor em contratos de adesão.",
    ],
    "DireitoCivil": [
        "O prazo prescricional para ações de reparação civil extracontratual é de três anos.",
        "A responsabilidade civil do Estado por ato omissivo depende da comprovação de culpa.",
        "O condomínio tem legitimidade para propor ação de cobrança de débito condominial.",
    ],
    "DireitoProcessual": [
        "É cabível mandado de segurança contra ato judicial desprovido de recurso próprio com efeito suspensivo.",
        "O juiz pode conhecer de ofício a prescrição nos feitos não patrimoniais.",
        "A reconvenção é cabível nos processos de execução e cautelar quando houver conexão com a ação principal.",
    ],
    "DireitoContratual": [
        "Os contratos bancários estão sujeitos ao Código de Defesa do Consumidor, sendo nulas as cláusulas abusivas.",
        "A cláusula penal não pode ser exigida simultaneamente com o cumprimento da obrigação principal.",
        "O fiador pode opor ao credor as exceções pessoais do devedor afiançado decorrentes do contrato.",
    ],
    "DireitoAmbiental": [
        "A responsabilidade por dano ambiental é objetiva e solidária, não se admitindo a denunciação da lide.",
        "A obrigação de reparar o dano ambiental é propter rem e transmite-se ao adquirente do imóvel.",
        "O licenciamento ambiental é obrigatório para atividades potencialmente causadoras de degradação.",
    ],
    "DireitoTrabalhista": [
        "O prazo prescricional trabalhista para ações contra a Fazenda Pública é quinquenal.",
        "A jornada de trabalho em regime de compensação deve ser estabelecida por acordo coletivo.",
        "É devido o adicional de insalubridade ao empregado exposto a agentes nocivos acima dos limites legais.",
    ],
    "DireitoConstitucional": [
        "É inconstitucional a exigência de depósito prévio como condição de admissibilidade de ação judicial.",
        "O mandado de segurança coletivo pode ser impetrado por partido político com representação no Congresso Nacional.",
        "A imunidade tributária recíproca veda a instituição de impostos sobre o patrimônio, renda ou serviços uns dos outros.",
    ],
}


@dataclass
class ClusterStats:
    """
    Aggregated statistics for a single cluster.

    Attributes:
        cluster_id: HDBSCAN cluster label
        case_type: Post-hoc case type assigned via BERTimbau seed embedding cosine similarity
        frequency: Number of sentences in this cluster
        representative_sentence: Sentence closest to the centroid
        non_representative_sentence: Sentence farthest from the centroid
        intra_similarity: Mean cosine similarity among cluster members
        centroid: Mean embedding vector for this cluster
    """

    cluster_id: int
    case_type: str
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
    cosine similarity. Case type labels are assigned by comparing the cluster
    centroid to pre-computed BERTimbau seed embedding cosine similarity for
    each case type. Cross-cluster cosine similarity between centroids
    reveals thematic overlap. Noise sentences (cluster_id == -1) are
    excluded from cluster statistics.
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize statistics generator.

        Args:
            model_name: HuggingFace model identifier for BERTimbau
            similarity_threshold: Minimum cosine similarity to assign a named case type; below this returns Unknown
        """
        super().__init__(
            step_number=8,
            name="Statistics Generator",
            description="Compute frequency tables, similarity, and post-hoc case type labels",
        )
        self._similarity_threshold = similarity_threshold
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._seed_embeddings = self._embed_seeds()

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

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string using BERTimbau mean pooling.

        Args:
            text: Input text to embed

        Returns:
            768-dim numpy array
        """
        tokens = self._tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**tokens)
        hidden: torch.Tensor = output.last_hidden_state
        mask: torch.Tensor = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return (summed / counts).squeeze(0).numpy()

    def _embed_seeds(self) -> dict[str, np.ndarray]:
        """
        Compute mean seed embedding per case type from _CASE_SEEDS.

        Returns:
            Dictionary mapping case type name to mean 768-dim embedding
        """
        result: dict[str, np.ndarray] = {}
        for case_type, seeds in _CASE_SEEDS.items():
            seed_vectors = np.stack([self._embed_text(seed) for seed in seeds])
            result[case_type] = seed_vectors.mean(axis=0)
        return result

    def _label_cluster_by_embedding(self, cluster_centroid: np.ndarray) -> str:
        """
        Assign a case type by cosine similarity between cluster centroid and seed embeddings.

        Args:
            cluster_centroid: Mean embedding vector for the cluster

        Returns:
            Case type name with highest similarity above threshold, or Unknown
        """
        centroid_2d = cluster_centroid.reshape(1, -1)
        best_case_type = "Unknown"
        best_similarity = -1.0
        for case_type, seed_embedding in self._seed_embeddings.items():
            similarity = float(cosine_similarity(centroid_2d, seed_embedding.reshape(1, -1))[0, 0])
            if similarity > best_similarity:
                best_similarity = similarity
                best_case_type = case_type
        if best_similarity >= self._similarity_threshold:
            return best_case_type
        return "Unknown"

    def _compute_cluster_stats(self, cluster_id: int, members: list) -> ClusterStats:
        """
        Compute statistics for one cluster and assign case type via seed embedding similarity.

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
        non_representative = members[int(np.argmin(dists))].text
        intra_sim = float(np.mean(cosine_similarity(embeddings)))
        case_type = self._label_cluster_by_embedding(centroid)
        return ClusterStats(
            cluster_id=cluster_id,
            case_type=case_type,
            frequency=len(members),
            representative_sentence=representative,
            non_representative_sentence=non_representative,
            intra_similarity=round(intra_sim, 4),
            centroid=centroid,
        )

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
