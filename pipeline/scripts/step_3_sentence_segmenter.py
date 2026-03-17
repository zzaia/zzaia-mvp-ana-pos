"""Step 3: Súmula-level segmentation using BOILERPLATE_TOKEN boundary splitting."""

from dataclasses import dataclass
from typing import Optional

from pipeline_step import PipelineStep
from step_2_boilerplate_remover import BOILERPLATE_TOKEN, BoilerplateOutput


@dataclass
class SegmentationOutput:
    """
    Output of the sentence segmentation step.

    Attributes:
        sentences: Ordered list of súmula block strings
        sentence_count: Total number of segments extracted
        source_path: Propagated from previous step
    """

    sentences: list[str]
    sentence_count: int
    source_path: Optional[object] = None


class SentenceSegmenter(PipelineStep):
    """
    Segment legal text into súmula-level units.

    Splits the filtered text on BOILERPLATE_TOKEN boundaries, where each
    token marks the position of the repeated page header that separates
    consecutive súmulas. Falls back to double-newline paragraph splitting
    when fewer than two boilerplate-delimited segments are found. Segments
    shorter than min_tokens words are discarded.
    """

    def __init__(self, min_tokens: int = 5):
        """
        Initialize sentence segmenter.

        Args:
            min_tokens: Minimum word count for a segment to be retained
        """
        super().__init__(
            step_number=3,
            name="Sentence Segmenter",
            description="Split document into súmula-level segments on BOILERPLATE_TOKEN boundaries",
        )
        self._min_tokens = min_tokens

    def _strip_boilerplate(self, text: str) -> str:
        """
        Remove boilerplate tokens from a text block.

        Args:
            text: Raw segment text possibly containing BOILERPLATE_TOKEN

        Returns:
            Text with all BOILERPLATE_TOKEN occurrences removed and stripped
        """
        return text.replace(BOILERPLATE_TOKEN, "").strip()

    def _split_on_boilerplate(self, text: str) -> list[str]:
        """
        Split text on BOILERPLATE_TOKEN boundaries into súmula segments.

        Args:
            text: Full document text with BOILERPLATE_TOKEN markers

        Returns:
            List of non-empty segment strings meeting min_tokens threshold
        """
        chunks = text.split(BOILERPLATE_TOKEN)
        segments: list[str] = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned.split()) >= self._min_tokens:
                segments.append(cleaned)
        return segments

    def _fallback_segment(self, text: str) -> list[str]:
        """
        Split text on double newlines as a paragraph-level fallback.

        Args:
            text: Full document text without boilerplate boundaries

        Returns:
            List of non-empty paragraph strings meeting min_tokens threshold
        """
        paragraphs = text.split("\n\n")
        segments: list[str] = []
        for paragraph in paragraphs:
            cleaned = self._strip_boilerplate(paragraph)
            if cleaned and len(cleaned.split()) >= self._min_tokens:
                segments.append(cleaned)
        return segments

    def process(self, input_data: BoilerplateOutput) -> SegmentationOutput:
        """
        Segment filtered text into súmula blocks or paragraphs.

        Args:
            input_data: BoilerplateOutput with filtered_text

        Returns:
            SegmentationOutput with ordered segment list
        """
        text = input_data.filtered_text
        segments = self._split_on_boilerplate(text)
        if len(segments) < 2:
            segments = self._fallback_segment(text)
        return SegmentationOutput(
            sentences=segments,
            sentence_count=len(segments),
            source_path=input_data.source_path,
        )

    def validate(self, output_data: SegmentationOutput) -> bool:
        """
        Validate that segmentation produced a meaningful segment set.

        Args:
            output_data: SegmentationOutput to validate

        Returns:
            True if sentence count is at least 10
        """
        return output_data.sentence_count >= 10
