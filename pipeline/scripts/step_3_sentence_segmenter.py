"""Step 3: Sentence segmentation using spaCy pt_core_news_lg with legal rules."""

from dataclasses import dataclass
from statistics import mean
from typing import Optional

import spacy
from spacy.language import Language

from pipeline_step import PipelineStep
from step_2_boilerplate_remover import BoilerplateOutput

_LEGAL_ABBREVS: list[str] = [
    "art", "arts", "inc", "par", "§", "cf", "v", "vide", "n", "n.o",
    "fls", "fl", "p", "pp", "ss", "obs", "ex", "dr", "dra", "des",
    "min", "proc", "prom", "coord", "adv", "mp", "pg", "pgs",
]


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

    Uses spaCy pt_core_news_lg with parser for accurate boundary detection,
    then applies post-processing to merge sentences incorrectly split after
    legal abbreviations. Sentences shorter than min_tokens are discarded.
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
        Load spaCy model lazily with parser for sentence boundary detection.

        Returns:
            Configured spaCy Language pipeline
        """
        if self._nlp is None:
            nlp = spacy.load("pt_core_news_lg", exclude=["ner", "lemmatizer"])
            nlp.max_length = self._max_length
            self._nlp = nlp
        return self._nlp

    def _merge_abbrev_sentences(self, sentences: list[str]) -> list[str]:
        """
        Merge sentences incorrectly split after legal abbreviations.

        Args:
            sentences: Raw sentence list from spaCy segmentation

        Returns:
            Sentence list with abbreviation-induced splits merged
        """
        merged: list[str] = []
        skip_next = False
        for i, sentence in enumerate(sentences):
            if skip_next:
                skip_next = False
                continue
            tokens = sentence.split()
            last_token = tokens[-1].lower().rstrip(".") if tokens else ""
            if last_token in _LEGAL_ABBREVS and i + 1 < len(sentences):
                merged.append(sentence + " " + sentences[i + 1])
                skip_next = True
            else:
                merged.append(sentence)
        return merged

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
        raw_sentences = [
            sent.text.strip()
            for sent in doc.sents
            if len(sent) >= self._min_tokens and sent.text.strip()
        ]
        sentences = self._merge_abbrev_sentences(raw_sentences)
        return SegmentationOutput(
            sentences=sentences,
            sentence_count=len(sentences),
            source_path=input_data.source_path,
        )

    def validate(self, output_data: SegmentationOutput) -> bool:
        """
        Validate that segmentation produced a meaningful sentence set.

        Args:
            output_data: SegmentationOutput to validate

        Returns:
            True if sentence count >= 10 and average sentence length < 5000
        """
        sentences = output_data.sentences
        if output_data.sentence_count < 10:
            return False
        if mean(len(s) for s in sentences) >= 5000:
            return False
        return True
