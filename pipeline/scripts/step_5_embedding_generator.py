"""Step 5: Sentence embedding using BERTimbau with mean pooling and sliding window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline_step import PipelineStep
from step_4_citation_normalizer import CitationOutput

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

@dataclass
class EmbeddedSentence:
    """
    A sentence augmented with its embedding vector.

    Attributes:
        text: Sentence text
        embedding: Float32 embedding vector from Legal-BERTimbau
        citation_metadata: Original citations
    """

    text: str
    embedding: np.ndarray
    citation_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingOutput:
    """
    Output of the embedding generation step.

    Attributes:
        embedded_sentences: Sentences with 768-dim embedding vectors
        model_name: HuggingFace model identifier used
        embedding_dim: Dimensionality of each embedding vector
        source_path: Propagated from previous step
    """

    embedded_sentences: list[EmbeddedSentence]
    model_name: str
    embedding_dim: int
    source_path: Optional[object] = None


class EmbeddingGenerator(PipelineStep):
    """
    Generate sentence embeddings with Legal-BERTimbau mean pooling.

    Uses rufimelo/Legal-BERTimbau-sts-large-ma fine-tuned on legal Portuguese
    semantic textual similarity tasks. Mean pooling over all non-padding token
    states produces the final vector. Sentences exceeding 512 tokens are handled
    via a sliding window with 64-token overlap; the window embeddings are averaged.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        batch_size: int = 16,
    ):
        """
        Initialize embedding generator.

        Args:
            max_tokens: BERT context window size
            overlap_tokens: Token overlap between adjacent windows
            batch_size: Number of sentences to encode at once
            model_name: HuggingFace model identifier to load
        """
        super().__init__(
            step_number=5,
            name="Embedding Generator",
            description="Generate embeddings with Legal-BERTimbau mean pooling",
        )
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._batch_size = batch_size
        self._model_name = model_name
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[PreTrainedModel] = None

    def _load_model(self) -> None:
        """Load tokenizer and model lazily using configured model name."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.eval()

    def process(self, input_data: CitationOutput) -> EmbeddingOutput:
        """
        Embed all sentences from citation normalization output.

        Args:
            input_data: CitationOutput with sentences and citation_metadata

        Returns:
            EmbeddingOutput with 768-dim embedding per sentence
        """
        self._load_model()
        embedded: list[EmbeddedSentence] = []
        sentences = list(zip(input_data.sentences, input_data.citation_metadata))
        for i in range(0, len(sentences), self._batch_size):
            batch = sentences[i: i + self._batch_size]
            texts = [text for text, _ in batch]
            vectors = self._embed_batch(texts)
            for (text, citations), vector in zip(batch, vectors):
                embedded.append(EmbeddedSentence(
                    text=text,
                    embedding=vector,
                    citation_metadata=citations,
                ))
        embedding_dim = embedded[0].embedding.shape[0] if embedded else 0
        return EmbeddingOutput(
            embedded_sentences=embedded,
            model_name=self._model_name,
            embedding_dim=embedding_dim,
            source_path=input_data.source_path,
        )

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode a batch of sentence texts into embedding vectors.

        Args:
            texts: List of sentence strings

        Returns:
            List of 768-dim numpy arrays in the same order
        """
        vectors: list[np.ndarray] = []
        for text in texts:
            vector = self._embed_single(text)
            vectors.append(vector)
        return vectors

    def _embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single sentence using sliding window if necessary.

        Args:
            text: Sentence text

        Returns:
            768-dim mean-pooled embedding
        """
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids: torch.Tensor = tokens["input_ids"][0]
        if len(input_ids) <= self._max_tokens:
            return self._mean_pool(tokens)
        return self._sliding_window(input_ids)

    def _mean_pool(self, tokens: dict) -> np.ndarray:
        """
        Compute mean-pooled embedding for tokenized input.

        Args:
            tokens: Tokenizer output dict

        Returns:
            768-dim numpy array
        """
        with torch.no_grad():
            output = self._model(**tokens)
        hidden: torch.Tensor = output.last_hidden_state
        mask: torch.Tensor = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return (summed / counts).squeeze(0).numpy()

    def _sliding_window(self, input_ids: torch.Tensor) -> np.ndarray:
        """
        Process a long sequence via overlapping windows and average the results.

        Args:
            input_ids: Full token id sequence

        Returns:
            768-dim averaged embedding
        """
        stride = self._max_tokens - self._overlap_tokens
        window_embeddings: list[np.ndarray] = []
        for start in range(0, len(input_ids), stride):
            chunk = input_ids[start: start + self._max_tokens]
            if len(chunk) < 4:
                break
            chunk_tokens = {
                "input_ids": chunk.unsqueeze(0),
                "attention_mask": torch.ones(1, len(chunk), dtype=torch.long),
            }
            window_embeddings.append(self._mean_pool(chunk_tokens))
        return np.mean(window_embeddings, axis=0)

    def validate(self, output_data: EmbeddingOutput) -> bool:
        """
        Validate that embeddings were generated with consistent dimensionality.

        Args:
            output_data: EmbeddingOutput to validate

        Returns:
            True if all embeddings have the expected shape
        """
        if not output_data.embedded_sentences:
            return False
        return all(
            item.embedding.shape == (output_data.embedding_dim,)
            for item in output_data.embedded_sentences
        )
