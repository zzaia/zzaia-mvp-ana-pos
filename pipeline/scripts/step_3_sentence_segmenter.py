"""Step 3: Sentence segmentation using spaCy pt_core_news_lg with legal rules."""

from dataclasses import dataclass, field
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from pipeline_step import PipelineStep
from step_2_boilerplate_remover import BoilerplateOutput

_LEGAL_ABBREVS: list[str] = [
    "art", "arts", "inc", "par", "§", "cf", "v", "vide", "n", "n.o",
    "fls", "fl", "p", "pp", "ss", "obs", "ex", "dr", "dra", "des",
    "min", "proc", "prom", "coord", "adv", "mp", "pg", "pgs",
]


@Language.component("legal_sentence_fixer")
def _legal_sentence_fixer(doc: Doc) -> Doc:
    """
    Prevent sentence breaks after common legal abbreviations.

    Args:
        doc: spaCy Doc object

    Returns:
        Doc with corrected sentence boundaries
    """
    for token in doc[:-1]:
        if token.text.lower().rstrip(".") in _LEGAL_ABBREVS:
            doc[token.i + 1].is_sent_start = False
    return doc


@dataclass
class SegmentationOutput:
    """
    Output of the sentence segmentation step.

    Attributes:
        sentences: Ordered list of sentence strings
        sentence_count: Total number of sentences extracted
        source_path: Propagated from previous step
    """

    sentences: list[str]
    sentence_count: int
    source_path: Optional[object] = None


class SentenceSegmenter(PipelineStep):
    """
    Segment legal text into ordered sentences.

    Uses spaCy pt_core_news_lg as the base segmenter and adds a custom
    pipeline component that prevents sentence splits after common Brazilian
    legal abbreviations. Sentences shorter than min_tokens are discarded
    to remove noise fragments.
    """

    def __init__(self, min_tokens: int = 5, max_length: int = 20000000):
        """
        Initialize sentence segmenter.

        Args:
            min_tokens: Minimum token count for a sentence to be retained
            max_length: Maximum character length for the spaCy pipeline
        """
        super().__init__(
            step_number=3,
            name="Sentence Segmenter",
            description="Split document into ordered sentences using spaCy legal rules",
        )
        self._max_length = max_length
        self._min_tokens = min_tokens
        self._nlp: Optional[Language] = None

    def _get_nlp(self) -> Language:
        """
        Load spaCy model lazily with the legal sentence fixer component.

        Returns:
            Configured spaCy Language pipeline
        """
        if self._nlp is None:
            nlp = spacy.load("pt_core_news_lg", disable=["ner", "lemmatizer"])
            nlp.max_length = self._max_length
            if "legal_sentence_fixer" not in nlp.pipe_names:
                nlp.add_pipe("legal_sentence_fixer", after="senter")
            self._nlp = nlp
        return self._nlp

    def process(self, input_data: BoilerplateOutput) -> SegmentationOutput:
        """
        Segment filtered text into sentences.

        Args:
            input_data: BoilerplateOutput with filtered_text

        Returns:
            SegmentationOutput with ordered sentence list
        """
        nlp = self._get_nlp()
        doc = nlp(input_data.filtered_text)
        sentences = [
            sent.text.strip()
            for sent in doc.sents
            if len(sent) >= self._min_tokens and sent.text.strip()
        ]
        return SegmentationOutput(
            sentences=sentences,
            sentence_count=len(sentences),
            source_path=input_data.source_path,
        )

    def validate(self, output_data: SegmentationOutput) -> bool:
        """
        Validate that at least one sentence was extracted.

        Args:
            output_data: SegmentationOutput to validate

        Returns:
            True if sentences list is non-empty
        """
        return len(output_data.sentences) > 0
