"""Step 10: FAISS semantic search index over BERTimbau-embedded Súmulas."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from step_5_embedding_generator import EmbeddingOutput
from step_6_bertopic import BerTopicOutput


@dataclass
class SearchResult:
    """
    A single semantic search result with topic context.

    Attributes:
        rank: 1-based result rank
        text: Súmula text content
        similarity: Cosine similarity to query (0-1)
        topic_id: BERTopic topic this sentence belongs to
        topic_label: Auto-generated topic keyword label
    """

    rank: int
    text: str
    similarity: float
    topic_id: int
    topic_label: str


class SumulaSearchIndex:
    """
    FAISS-based semantic search index over BERTimbau-embedded Súmulas.

    Builds an inner-product index from pre-computed embeddings.
    At search time, embeds the query with BERTimbau and returns
    the top-k most semantically similar Súmulas with topic context.
    """

    def __init__(
        self,
        embedding_output: EmbeddingOutput,
        bertopic_output: BerTopicOutput,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
    ):
        """
        Build FAISS index from pre-computed embeddings and load query model.

        Args:
            embedding_output: Step 5 output with pre-computed sentence embeddings
            bertopic_output: Step 6 output with topic assignments and labels
            model_name: HuggingFace model for query embedding
        """
        self._sentences = [s.text for s in embedding_output.embedded_sentences]
        self._topic_lookup: dict[str, tuple[int, str]] = {}
        for s in bertopic_output.topiced_sentences:
            label = bertopic_output.topic_labels.get(s.topic_id, str(s.topic_id))
            self._topic_lookup[s.text] = (s.topic_id, label)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        embeddings = np.stack([s.embedding for s in embedding_output.embedded_sentences])
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
            embeddings: (n, 768) raw embedding matrix

        Returns:
            Populated faiss.IndexFlatIP ready for search
        """
        normalized = self._normalize(embeddings)
        index = faiss.IndexFlatIP(768)
        index.add(normalized)
        return index

    def _embed_query(self, text: str) -> np.ndarray:
        """
        Tokenize and mean-pool query text through BERTimbau, then L2-normalize.

        Args:
            text: Portuguese legal query text

        Returns:
            L2-normalized 768-dim float32 vector shaped (1, 768)
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
            text = self._sentences[idx]
            topic_id, topic_label = self._topic_lookup.get(text, (-1, "unknown"))
            results.append(SearchResult(
                rank=rank,
                text=text,
                similarity=float(score),
                topic_id=topic_id,
                topic_label=topic_label,
            ))
        return results
