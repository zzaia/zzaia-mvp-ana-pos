"""Step 7: UMAP dimensionality reduction per rhetorical role group."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import umap

from pipeline_step import PipelineStep
from step_6_embedding_generator import EmbeddedSentence, EmbeddingOutput


@dataclass
class ReducedSentence:
    """
    A sentence with its UMAP-reduced vector.

    Attributes:
        text: Sentence text
        role: Rhetorical role label
        confidence: Role classification confidence
        embedding: Original 768-dim embedding
        reduced_vector: UMAP-reduced n_components-dim vector
        citation_metadata: Original citations
    """

    text: str
    role: str
    confidence: float
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

    UMAP is applied independently per rhetorical role group so that the
    topology learned within each group is not distorted by inter-role
    distances. Using cosine metric preserves semantic similarity for
    high-dimensional normalized BERT embeddings. A fixed random_state
    ensures reproducibility.
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
            step_number=7,
            name="UMAP Reducer",
            description="Reduce 768-dim embeddings to n_components per role group",
        )
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._random_state = random_state

    def process(self, input_data: EmbeddingOutput) -> UmapOutput:
        """
        Apply UMAP reduction grouped by rhetorical role.

        Args:
            input_data: EmbeddingOutput with 768-dim embeddings

        Returns:
            UmapOutput with reduced vectors per sentence
        """
        sentences = input_data.embedded_sentences
        role_groups: dict[str, list[int]] = {}
        for idx, item in enumerate(sentences):
            role_groups.setdefault(item.role, []).append(idx)
        reduced_map: dict[int, np.ndarray] = {}
        for role, indices in role_groups.items():
            matrix = np.stack([sentences[i].embedding for i in indices])
            reduced = self._reduce_group(matrix)
            for local_idx, global_idx in enumerate(indices):
                reduced_map[global_idx] = reduced[local_idx]
        reduced_sentences = [
            ReducedSentence(
                text=item.text,
                role=item.role,
                confidence=item.confidence,
                embedding=item.embedding,
                reduced_vector=reduced_map[idx],
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

    def _reduce_group(self, matrix: np.ndarray) -> np.ndarray:
        """
        Fit and transform a matrix of embeddings with UMAP.

        Args:
            matrix: (n_samples, 768) embedding matrix for one role group

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
