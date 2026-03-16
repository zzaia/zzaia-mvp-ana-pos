"""Step 6: UMAP dimensionality reduction on all sentences in a single projection."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import umap

from pipeline_step import PipelineStep
from step_5_embedding_generator import EmbeddingOutput


@dataclass
class ReducedSentence:
    """
    A sentence with its UMAP-reduced vector.

    Attributes:
        text: Sentence text
        embedding: Original 768-dim embedding
        reduced_vector: UMAP-reduced n_components-dim vector
        citation_metadata: Original citations
    """

    text: str
    embedding: np.ndarray
    reduced_vector: np.ndarray
    citation_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class UmapOutput:
    """
    Output of the UMAP reduction step.

    Attributes:
        reduced_sentences: Sentences with low-dimensional vectors
        n_components: Target UMAP dimensionality
        n_neighbors: UMAP neighbors parameter used
        metric: UMAP distance metric used
        source_path: Propagated from previous step
    """

    reduced_sentences: list[ReducedSentence]
    n_components: int
    n_neighbors: int
    metric: str
    source_path: Optional[object] = None


class UmapReducer(PipelineStep):
    """
    Reduce 768-dim embeddings to a lower-dimensional space using UMAP.

    A single UMAP projection is applied to all sentences simultaneously,
    preserving the global topology across the entire corpus. Cosine metric
    preserves semantic similarity for high-dimensional normalised BERT
    embeddings. A fixed random_state ensures reproducibility.
    """

    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        metric: str = "cosine",
        random_state: int = 42,
    ):
        """
        Initialize UMAP reducer.

        Args:
            n_components: Target dimensionality for UMAP output
            n_neighbors: UMAP local neighborhood size
            metric: Distance metric for UMAP
            random_state: Seed for reproducibility
        """
        super().__init__(
            step_number=6,
            name="UMAP Reducer",
            description="Reduce 768-dim embeddings to n_components in a single projection",
        )
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._random_state = random_state

    def process(self, input_data: EmbeddingOutput) -> UmapOutput:
        """
        Apply UMAP reduction to all sentences in one single projection.

        Args:
            input_data: EmbeddingOutput with 768-dim embeddings

        Returns:
            UmapOutput with reduced vectors per sentence
        """
        sentences = input_data.embedded_sentences
        matrix = np.stack([item.embedding for item in sentences])
        reduced = self._reduce(matrix)
        reduced_sentences = [
            ReducedSentence(
                text=item.text,
                embedding=item.embedding,
                reduced_vector=reduced[idx],
                citation_metadata=item.citation_metadata,
            )
            for idx, item in enumerate(sentences)
        ]
        return UmapOutput(
            reduced_sentences=reduced_sentences,
            n_components=self._n_components,
            n_neighbors=self._n_neighbors,
            metric=self._metric,
            source_path=input_data.source_path,
        )

    def _reduce(self, matrix: np.ndarray) -> np.ndarray:
        """
        Fit and transform the full embedding matrix with UMAP.

        Args:
            matrix: (n_samples, 768) embedding matrix for all sentences

        Returns:
            (n_samples, n_components) reduced matrix
        """
        effective_neighbors = min(self._n_neighbors, len(matrix) - 1)
        if effective_neighbors < 2:
            return np.zeros((len(matrix), self._n_components), dtype=np.float32)
        reducer = umap.UMAP(
            n_components=self._n_components,
            n_neighbors=effective_neighbors,
            metric=self._metric,
            random_state=self._random_state,
        )
        return reducer.fit_transform(matrix)

    def validate(self, output_data: UmapOutput) -> bool:
        """
        Validate reduced vectors have the expected dimensionality.

        Args:
            output_data: UmapOutput to validate

        Returns:
            True if all reduced vectors have shape (n_components,)
        """
        if not output_data.reduced_sentences:
            return False
        return all(
            item.reduced_vector.shape == (output_data.n_components,)
            for item in output_data.reduced_sentences
        )
