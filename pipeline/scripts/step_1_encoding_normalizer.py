"""Step 1: Encoding normalization using ftfy and Unicode NFC form."""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ftfy

from pipeline_step import PipelineStep
from step_0_pdf_reader import PdfReaderOutput


@dataclass
class NormalizerOutput:
    """
    Output of the encoding normalizer step.

    Attributes:
        clean_text: UTF-8 NFC normalized text
        replacements: Log of (original, fixed) pairs applied by ftfy
        source_path: Propagated from previous step
    """

    clean_text: str
    replacements: list[tuple[str, str]] = field(default_factory=list)
    source_path: Optional[Path] = None


class EncodingNormalizer(PipelineStep):
    """
    Fix encoding errors and normalize to UTF-8 NFC.

    Applies ftfy to repair mojibake and other encoding artifacts,
    then runs Unicode NFC normalization to unify character representations.
    All replacements are logged for auditability.
    """

    def __init__(self):
        """Initialize encoding normalizer."""
        super().__init__(
            step_number=1,
            name="Encoding Normalizer",
            description="Fix encoding errors and normalize to UTF-8 NFC",
        )

    def process(self, input_data: PdfReaderOutput) -> NormalizerOutput:
        """
        Apply ftfy and NFC normalization to raw text.

        Args:
            input_data: PdfReaderOutput containing raw_text

        Returns:
            NormalizerOutput with clean_text and replacement log
        """
        raw = input_data.raw_text
        fixed = ftfy.fix_text(raw)
        replacements = self._log_replacements(raw, fixed)
        normalized = unicodedata.normalize("NFC", fixed)
        return NormalizerOutput(
            clean_text=normalized,
            replacements=replacements,
            source_path=input_data.source_path,
        )

    def _log_replacements(self, original: str, fixed: str) -> list[tuple[str, str]]:
        """
        Identify and log token-level replacements made by ftfy using difflib.

        Uses difflib.ndiff to compare word sequences so that insertions,
        deletions, and replacements are all captured correctly, regardless
        of length differences between the two texts.

        Args:
            original: Text before ftfy processing
            fixed: Text after ftfy processing

        Returns:
            List of (original_fragment, fixed_fragment) tuples where they differ
        """
        if original == fixed:
            return []
        orig_words = original.split()
        fixed_words = fixed.split()
        replacements: list[tuple[str, str]] = []
        removed: list[str] = []
        added: list[str] = []
        for tag in difflib.ndiff(orig_words, fixed_words):
            code = tag[:2]
            token = tag[2:]
            if code == "- ":
                removed.append(token)
            elif code == "+ ":
                added.append(token)
            else:
                if removed or added:
                    replacements.append((" ".join(removed), " ".join(added)))
                removed.clear()
                added.clear()
        if removed or added:
            replacements.append((" ".join(removed), " ".join(added)))
        return replacements

    def validate(self, output_data: NormalizerOutput) -> bool:
        """
        Validate that clean text is non-empty.

        Args:
            output_data: NormalizerOutput to validate

        Returns:
            True if clean_text is non-empty
        """
        return bool(output_data.clean_text.strip())
