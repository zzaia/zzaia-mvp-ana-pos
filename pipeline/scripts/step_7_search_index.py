"""Step 7: FAISS semantic search index over labeled BERTimbau-embedded Súmulas."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from step_6_embedding_generator import EmbeddingOutput


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
