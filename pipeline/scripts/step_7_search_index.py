"""Step 7: FAISS semantic search index over labeled BERTimbau-embedded Súmulas."""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline_step import PipelineStep
from step_6_embedding_generator import EmbeddingOutput, EmbeddedSentence


@dataclass
class SearchResult:
    """
    A single semantic search result with label metadata.

    Attributes:
        rank: 1-based result position
        text: Súmula text content
        similarity: Cosine similarity to query (0–1)
        label: Composite label from the labeling step
        sumula_number: Numeric súmula identifier
    """

    rank: int
    text: str
    similarity: float
    label: str
    sumula_number: int

    @property
    def area(self) -> str:
        """Legal area extracted from the composite label prefix."""
        return self.label.split("_")[0]


@dataclass
class SearchIndexOutput:
    """
    Output of the search index step.

    Attributes:
        index: Populated SumulaSearchIndex ready for queries
        results: Search results for the configured query
        query: Query string used to produce results
        embedded_sentences: Propagated embedded sentences for downstream use
        accumulated_similarity: Sum of cosine similarities across all results (computed property)
    """

    index: SumulaSearchIndex
    results: list[SearchResult]
    query: str
    embedded_sentences: list[EmbeddedSentence] = field(default_factory=list)

    @property
    def accumulated_similarity(self) -> float:
        """Sum of cosine similarities across all search results."""
        return sum(r.similarity for r in self.results)

    def accumulated_similarity_above(self, threshold: float) -> float:
        """
        Sum of cosine similarities for results exceeding a threshold.

        Args:
            threshold: Minimum cosine similarity to include in the sum

        Returns:
            Sum of similarities above the threshold
        """
        return sum(r.similarity for r in self.results if r.similarity >= threshold)

    @property
    def excess_similarity(self) -> float:
        """Sum of above-mean similarity deviations across all results."""
        if not self.results:
            return 0.0
        mean_sim = self.accumulated_similarity / len(self.results)
        return sum(r.similarity - mean_sim for r in self.results if r.similarity > mean_sim)

    def precision_ratio(self, threshold: float) -> float:
        """
        Ratio of accumulated similarity above threshold to total accumulated similarity.

        Args:
            threshold: Minimum cosine similarity cutoff

        Returns:
            Ratio in [0, 1]; higher means results concentrate above threshold
        """
        total = self.accumulated_similarity
        if total == 0.0:
            return 0.0
        return self.accumulated_similarity_above(threshold) / total

    @property
    def area_similarities(self) -> dict[str, float]:
        """Accumulated cosine similarity grouped by legal area label prefix."""
        groups: dict[str, float] = {}
        for r in self.results:
            groups[r.area] = groups.get(r.area, 0.0) + r.similarity
        return groups


class SumulaSearchIndex:
    """
    FAISS semantic search index over labeled Súmula embeddings.

    Builds a faiss.IndexFlatIP from L2-normalized pre-computed embeddings.
    At search time, encodes the Portuguese query with the same BERTimbau model
    and returns the top-k most semantically similar Súmulas with label context.
    """

    def __init__(self, embedding_output: EmbeddingOutput):
        """
        Build FAISS index from pre-computed embeddings and load the query model.

        Args:
            embedding_output: Step 6 output with pre-computed sentence embeddings
        """
        self._sentences = embedding_output.embedded_sentences
        self._tokenizer = AutoTokenizer.from_pretrained(embedding_output.model_name)
        self._model = AutoModel.from_pretrained(embedding_output.model_name)
        self._model.eval()
        embeddings = np.stack([s.embedding for s in self._sentences])
        self._index = self._build_index(embeddings)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize each row of the embedding matrix.

        Args:
            vectors: (n, d) embedding matrix

        Returns:
            L2-normalized (n, d) matrix as float32
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def _build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create and populate a FAISS inner-product index from normalized embeddings.

        Args:
            embeddings: (n, d) raw embedding matrix

        Returns:
            Populated faiss.IndexFlatIP ready for search
        """
        normalized = self._normalize(embeddings)
        dim = normalized.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)
        return index

    def _embed_query(self, text: str) -> np.ndarray:
        """
        Tokenize and mean-pool query text through BERTimbau, then L2-normalize.

        Args:
            text: Portuguese legal query text

        Returns:
            L2-normalized float32 vector shaped (1, d)
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
        vector = (summed / counts).squeeze(0).numpy()
        return self._normalize(vector.reshape(1, -1))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve the top-k semantically similar Súmulas for a Portuguese query.

        Args:
            query: Portuguese legal text to search for
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by similarity descending
        """
        query_vec = self._embed_query(query)
        scores, indices = self._index.search(query_vec, top_k)
        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._sentences):
                continue
            sentence = self._sentences[idx]
            results.append(SearchResult(
                rank=rank,
                text=sentence.text,
                similarity=float(score),
                label=sentence.label,
                sumula_number=sentence.sumula_number,
            ))
        return results


class SearchIndexBuilder(PipelineStep):
    """
    Build a FAISS semantic search index and execute one or more queries.

    Accepts a single query string or a list of queries. Returns a list of
    SearchIndexOutput, one per query, all sharing the same index instance.
    """

    def __init__(self, queries: list[str] | str, top_k: int = 288):
        """
        Initialize search index builder.

        Args:
            queries: One or more Portuguese legal queries to run against the index
            top_k: Number of results to retrieve per query
        """
        super().__init__(
            step_number=7,
            name="Search Index",
            description="Build FAISS index and execute semantic search queries",
        )
        self._queries = [queries] if isinstance(queries, str) else queries
        self._top_k = top_k

    def process(self, input_data: EmbeddingOutput) -> list[SearchIndexOutput]:
        """
        Build index once, run all queries, and return one output per query.

        Args:
            input_data: EmbeddingOutput from step 6

        Returns:
            List of SearchIndexOutput, one per configured query
        """
        index = SumulaSearchIndex(embedding_output=input_data)
        return [
            SearchIndexOutput(
                index=index,
                results=index.search(query, top_k=self._top_k),
                query=query,
                embedded_sentences=input_data.embedded_sentences,
            )
            for query in self._queries
        ]

    def validate(self, output_data: list[SearchIndexOutput]) -> bool:
        """
        Validate that every query produced at least one result.

        Args:
            output_data: List of SearchIndexOutput to validate

        Returns:
            True if all outputs have non-empty results
        """
        return bool(output_data) and all(len(o.results) > 0 for o in output_data)
