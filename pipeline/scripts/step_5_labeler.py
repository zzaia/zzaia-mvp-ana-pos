"""Step 5: Súmula labeler that extracts area, sub-area, and súmula number from segments."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from pipeline_step import PipelineStep
from step_4_citation_normalizer import CitationOutput

_SUMULA_PATTERN = re.compile(r"(?i)s[uú]mula\s+n?[.°]?\s*(\d+)")
_ANNOTATION_PATTERN = re.compile(r"(?i)^\(s[uú]mula\s+(cancelada|alterada)\)")


def _sanitize(text: str) -> str:
    """
    Strip accents and replace spaces with underscores, keeping alphanumeric and underscore.

    Args:
        text: Raw label component string

    Returns:
        Sanitized title-case string safe for use in label identifiers
    """
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    titled = normalized.title()
    words = re.sub(r"[^a-zA-Z0-9\s]", "", titled)
    return "_".join(words.split())


@dataclass
class LabeledSentence:
    """
    A súmula segment annotated with its legal area, sub-area, and súmula number.

    Attributes:
        text: Segment text with normalized citations
        label: Composite label as Area_Sub_area_N
        sumula_number: Numeric súmula identifier
        area: Legal area extracted from document header
        sub_area: Sub-area extracted from document header
        citation_metadata: Citation token mapping from step 4
    """

    text: str
    label: str
    sumula_number: int
    area: str
    sub_area: str
    citation_metadata: dict[str, str]


@dataclass
class LabeledOutput:
    """
    Output of the súmula labeling step.

    Attributes:
        labeled_sentences: All segments annotated with label metadata
        source_path: Propagated from previous step
    """

    labeled_sentences: list[LabeledSentence]
    source_path: Optional[object] = None


class SumulaLabeler(PipelineStep):
    """
    Label each súmula segment with area, sub-area, and súmula number.

    Extracts the first 1–3 non-empty lines before the súmula declaration line
    to derive legal area and sub-area categories. Falls back to 'Desconhecido'
    when header lines are absent. Súmula numbers are extracted via regex from
    the standard STJ format.
    """

    def __init__(self):
        """Initialize súmula labeler."""
        super().__init__(
            step_number=5,
            name="Sumula Labeler",
            description="Label segments with legal area, sub-area, and súmula number",
        )

    def _extract_sumula_number(self, text: str) -> Optional[int]:
        """
        Extract súmula number from segment text using standard pattern.

        Args:
            text: Segment text

        Returns:
            Integer súmula number or None if not found
        """
        match = _SUMULA_PATTERN.search(text)
        if match:
            return int(match.group(1))
        return None

    def _extract_header_lines(self, text: str) -> list[str]:
        """
        Extract the area/sub-area header line from a súmula segment.

        Handles two STJ layout variants:
        1. Standard: súmula number on line 0, header on line 1 (e.g. 'DIREITO BANCÁRIO - CONTRATO BANCÁRIO')
        2. Inline: header and súmula number merged on same line due to PDF extraction artifacts

        Annotation lines like '(SÚMULA CANCELADA)' are skipped.

        Args:
            text: Full segment text

        Returns:
            List with up to 1 header string after normalization
        """
        lines = text.splitlines()
        found_sumula = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if not found_sumula:
                if _SUMULA_PATTERN.search(stripped):
                    found_sumula = True
                    inline = re.sub(_SUMULA_PATTERN, "", stripped).strip(" -")
                    if re.search(r"DIREITO\s+\w", inline, re.IGNORECASE):
                        return [inline]
                continue
            if _ANNOTATION_PATTERN.match(stripped):
                continue
            return [stripped]
        return []

    def _build_label(self, area: str, sub_area: str, sumula_number: int) -> str:
        """
        Construct composite label from area, sub-area, and súmula number.

        Args:
            area: Sanitized legal area string
            sub_area: Sanitized sub-area string
            sumula_number: Integer súmula number

        Returns:
            Label string in format Area_Sub_area_N
        """
        return f"{area}_{sub_area}_{sumula_number}"

    def _label_segment(
        self, text: str, citation_metadata: dict[str, str], index: int
    ) -> LabeledSentence:
        """
        Produce a LabeledSentence from a single segment.

        Args:
            text: Segment text
            citation_metadata: Citation token mapping for this segment
            index: Fallback index when súmula number cannot be extracted

        Returns:
            LabeledSentence with all label fields populated
        """
        sumula_number = self._extract_sumula_number(text)
        header_lines = self._extract_header_lines(text)
        if sumula_number is None:
            sumula_number = index
        if header_lines:
            parts = header_lines[0].split(" - ", 1)
            area_raw = re.sub(r"(?i)^DIREITO\s+", "", parts[0]).strip()
            area = _sanitize(area_raw)
            sub_area = _sanitize(parts[1]) if len(parts) > 1 else "Desconhecido"
        else:
            area = "Desconhecido"
            sub_area = "Desconhecido"
        label = self._build_label(area, sub_area, sumula_number)
        return LabeledSentence(
            text=text,
            label=label,
            sumula_number=sumula_number,
            area=area,
            sub_area=sub_area,
            citation_metadata=citation_metadata,
        )

    def process(self, input_data: CitationOutput) -> LabeledOutput:
        """
        Label all súmula segments from citation normalization output.

        Args:
            input_data: CitationOutput with sentences and citation metadata

        Returns:
            LabeledOutput with all segments annotated
        """
        labeled: list[LabeledSentence] = []
        for index, (text, citations) in enumerate(
            zip(input_data.sentences, input_data.citation_metadata), start=1
        ):
            labeled.append(self._label_segment(text, citations, index))
        return LabeledOutput(
            labeled_sentences=labeled,
            source_path=input_data.source_path,
        )

    def validate(self, output_data: LabeledOutput) -> bool:
        """
        Validate that at least one labeled sentence was produced.

        Args:
            output_data: LabeledOutput to validate

        Returns:
            True if labeled_sentences list is non-empty
        """
        return len(output_data.labeled_sentences) > 0
