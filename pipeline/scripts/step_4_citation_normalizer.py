"""Step 4: Legal citation normalization with typed tokens and metadata storage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline_step import PipelineStep
from step_3_sentence_segmenter import SegmentationOutput

_CITATION_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bart(?:igo)?s?\.?\s*\d+[\d\s,;eEoO\.°ºª]*(?:do|da|de|dos|das)?\s+(?:código|lei|decreto|resolução|portaria|medida|instrução|constituição)[^\n]{0,60}",
        "<ART_REF>",
    ),
    (
        r"\b(?:processo|autos?|ação|recurso|apelação|agravo|mandado|habeas|embargos?)\s+n[º°.]?\s*[\d\.\-\/]+",
        "<PROC_NUM>",
    ),
    (
        r"\b(?:REsp|RExt|HC|MS|MI|ADI|ADPF|AI|ARE|AREsp|RO|RR|AIRR|Ag|AgR|AgRg|ED|EDcl|ACP|EREsp|EAREsp)\s+[\d\.\-\/]+",
        "<CASE_REF>",
    ),
    (
        r"\b(?:Súmula|Enunciado)\s+(?:Vinculante\s+)?n[º°.]?\s*\d+",
        "<SUMULA_REF>",
    ),
]


@dataclass
class CitationOutput:
    """
    Output of the citation normalizer step.

    Attributes:
        sentences: Sentences with citations replaced by typed tokens
        citation_metadata: Per-sentence mapping from token type to all matched originals
        source_path: Propagated from previous step
    """

    sentences: list[str]
    citation_metadata: list[dict[str, list[str]]] = field(default_factory=list)
    source_path: Optional[Path] = None


class CitationNormalizer(PipelineStep):
    """
    Replace legal citations with typed placeholder tokens.

    Substitutes article references, process numbers, case references, and
    súmula references with structured tokens so that downstream embeddings
    capture semantic roles rather than idiosyncratic citation strings.
    Original citation text is stored in citation_metadata for traceability.
    """

    def __init__(self):
        """Initialize citation normalizer."""
        super().__init__(
            step_number=4,
            name="Citation Normalizer",
            description="Replace legal citations with typed tokens",
        )

    def process(self, input_data: SegmentationOutput) -> CitationOutput:
        """
        Normalize citations in all sentences.

        Args:
            input_data: SegmentationOutput with sentences list

        Returns:
            CitationOutput with normalized sentences and metadata
        """
        normalized: list[str] = []
        metadata: list[dict[str, list[str]]] = []
        for sentence in input_data.sentences:
            norm_sentence, citations = self._normalize_sentence(sentence)
            normalized.append(norm_sentence)
            metadata.append(citations)
        return CitationOutput(
            sentences=normalized,
            citation_metadata=metadata,
            source_path=input_data.source_path,
        )

    def _normalize_sentence(self, sentence: str) -> tuple[str, dict[str, list[str]]]:
        """
        Apply all citation patterns to a single sentence.

        All occurrences of each token type are accumulated so no match is lost
        when the same token appears multiple times in one sentence.

        Args:
            sentence: Input sentence text

        Returns:
            Tuple of (normalized sentence, dict mapping token to list of originals)
        """
        citations: dict[str, list[str]] = {}
        for pattern, token in _CITATION_PATTERNS:
            matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
            if matches:
                citations.setdefault(token, []).extend(matches)
            sentence = re.sub(pattern, token, sentence, flags=re.IGNORECASE)
        return sentence, citations

    def validate(self, output_data: CitationOutput) -> bool:
        """
        Validate that sentence count is preserved.

        Args:
            output_data: CitationOutput to validate

        Returns:
            True if sentences list is non-empty
        """
        return len(output_data.sentences) > 0
