"""Step 2: Boilerplate removal using regex patterns and TF-IDF repetition detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_step import PipelineStep
from step_1_encoding_normalizer import NormalizerOutput

BOILERPLATE_TOKEN = "<BOILERPLATE>"

_HEADER_FOOTER_PATTERNS: list[str] = [
    r"(?i)[sú]?[úu]mulas\s+organizadas\s+por[\s\S]{0,20}?ramo\s+do\s+direito"
]

_REF_LINE_PATTERNS: list[str] = [
    r"(?im)^\s*(?:REP)?DJ\s+DATA:\d{2}/\d{2}/\d{4}.*$",
    r"(?im)^\s*R[A-Z]+STJ\s+VOL\.:\d+\s+PG:\d+.*$",
    r"(?im)^\s*R[A-Z]+\s+VOL\.:\d+\s+PG:\d+.*$",
    r"(?im)^\s*LEX[A-Z]+\s+VOL\.:\d+\s+PG:\d+.*$",
    r"(?im)^\s*scon\.stj\.jus\.br/.*$",
    r"(?im)^\s*\d{3}\.\d+S\d*\.\s*.*$",
    r"(?im)^\s*Fonte:\s*$",
    r"(?im)^\s*Referências Legislativas:\s*$",
    r"(?im)^\s*Precedentes:\s*$",
    r"(?i)(?:REP)?DJ\s+DATA:\d{2}/\d{2}/\d{4}\s+PG:\d+",
    r"(?i)R[A-Z]+STJ\s+VOL\.:\d+\s+PG:\d+",
    r"(?i)DEL:\d+\s+ANO:\d+\s+(?:REP)?DJ\s+DATA:[^\n]+",
]


@dataclass
class BoilerplateOutput:
    """
    Output of the boilerplate removal step.

    Attributes:
        filtered_text: Text with boilerplate replaced by BOILERPLATE_TOKEN
        removed_count: Number of segments replaced
        tfidf_threshold: Cosine similarity threshold used for repetition detection
        source_path: Propagated from previous step
    """

    filtered_text: str
    removed_count: int
    tfidf_threshold: float
    source_path: Optional[Path] = None


class BoilerplateRemover(PipelineStep):
    """
    Replace legal document boilerplate with a placeholder token.

    Combines deterministic regex patterns for common headers/footers
    with TF-IDF cosine similarity to detect repeated near-duplicate
    paragraphs across the document. Removed content is replaced with
    BOILERPLATE_TOKEN rather than deleted to preserve document structure.
    """

    def __init__(self, tfidf_threshold: float = 0.92, min_paragraph_tokens: int = 5):
        """
        Initialize boilerplate remover.

        Args:
            tfidf_threshold: Cosine similarity above which paragraphs are boilerplate
            min_paragraph_tokens: Minimum token count for TF-IDF consideration
        """
        super().__init__(
            step_number=2,
            name="Boilerplate Remover",
            description="Replace document boilerplate with placeholder tokens",
        )
        self._tfidf_threshold = tfidf_threshold
        self._min_paragraph_tokens = min_paragraph_tokens

    def process(self, input_data: NormalizerOutput) -> BoilerplateOutput:
        """
        Apply regex, TF-IDF, and reference line boilerplate removal.

        Args:
            input_data: NormalizerOutput with clean_text

        Returns:
            BoilerplateOutput with filtered_text and removal statistics
        """
        text = input_data.clean_text
        text, regex_count = self._apply_regex(text)
        text, tfidf_count = self._apply_tfidf(text)
        text, ref_count = self._strip_ref_lines(text)
        return BoilerplateOutput(
            filtered_text=text,
            removed_count=regex_count + tfidf_count + ref_count,
            tfidf_threshold=self._tfidf_threshold,
            source_path=input_data.source_path,
        )

    def _apply_regex(self, text: str) -> tuple[str, int]:
        """
        Replace regex-matched header/footer patterns.

        Args:
            text: Input text

        Returns:
            Tuple of (processed text, replacement count)
        """
        count = 0
        for pattern in _HEADER_FOOTER_PATTERNS:
            new_text, n = re.subn(pattern, BOILERPLATE_TOKEN, text)
            count += n
            text = new_text
        return text, count

    def _apply_tfidf(self, text: str) -> tuple[str, int]:
        """
        Detect and replace near-duplicate paragraphs via TF-IDF similarity.

        Args:
            text: Text after regex processing

        Returns:
            Tuple of (processed text, replacement count)
        """
        paragraphs = text.split("\n\n")
        eligible = [
            (i, p) for i, p in enumerate(paragraphs)
            if len(p.split()) >= self._min_paragraph_tokens
        ]
        if len(eligible) < 2:
            return text, 0
        indices, contents = zip(*eligible)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(contents)
        similarity: np.ndarray = cosine_similarity(matrix)
        np.fill_diagonal(similarity, 0.0)
        boilerplate_indices: set[int] = set()
        for row in range(len(contents)):
            if any(similarity[row] >= self._tfidf_threshold):
                boilerplate_indices.add(indices[row])
        count = 0
        for idx in boilerplate_indices:
            if paragraphs[idx] != BOILERPLATE_TOKEN:
                paragraphs[idx] = BOILERPLATE_TOKEN
                count += 1
        return "\n\n".join(paragraphs), count

    def _strip_ref_lines(self, text: str) -> tuple[str, int]:
        """
        Delete legislative reference lines from text without replacement tokens.

        Args:
            text: Text after TF-IDF processing

        Returns:
            Tuple of (processed text, count of matched lines removed)
        """
        count = 0
        for pattern in _REF_LINE_PATTERNS:
            matches = re.findall(pattern, text)
            count += len(matches)
            text = re.sub(pattern, "", text)
        return text, count

    def validate(self, output_data: BoilerplateOutput) -> bool:
        """
        Validate that filtered text is non-empty.

        Args:
            output_data: BoilerplateOutput to validate

        Returns:
            True if filtered_text has meaningful content beyond placeholders
        """
        content = output_data.filtered_text.replace(BOILERPLATE_TOKEN, "").strip()
        return bool(content)
