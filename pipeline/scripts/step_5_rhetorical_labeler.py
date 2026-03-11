"""Step 5: Rhetorical role labeling via keyword heuristics (placeholder for fine-tuned model)."""

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline_step import PipelineStep
from step_4_citation_normalizer import CitationOutput

ROLES = ("Fact", "Argument", "Ruling", "Precedent", "Procedural", "Uncertain")

_ROLE_PATTERNS: dict[str, list[str]] = {
    "Fact": [
        r"\b(ocorreu|ocorreram|constatou[-\s]se|verificou[-\s]se|comprovou[-\s]se|foi\s+comprovado|restou\s+demonstrado|dos\s+autos\s+consta|segundo\s+os\s+autos)\b",
        r"\b(na\s+data\s+de|em\s+\d{1,2}[\./]\d{1,2}[\./]\d{2,4}|o\s+réu|a\s+autora?|o\s+autor|o\s+reclamante)\b",
    ],
    "Argument": [
        r"\b(portanto|por(?:tanto|isso)|desta\s+forma|assim\s+sendo|em\s+razão\s+disso|nesse\s+sentido|logo|consequentemente|em\s+face\s+do\s+exposto)\b",
        r"\b(entende[-\s]se|verifica[-\s]se|demonstra[-\s]se|afirma|argumenta|sustenta|defende|alega)\b",
    ],
    "Ruling": [
        r"\b(decide[-\s]se|julga[-\s]se|condena[-\s]se|absolve[-\s]se|defere[-\s]se|indefere[-\s]se|determina[-\s]se|ordena[-\s]se)\b",
        r"\b(pelo\s+exposto|diante\s+do\s+exposto|ante\s+o\s+exposto|em\s+face\s+do\s+exposto|isto\s+posto|dispositivo)\b",
        r"\b(DECIDO|JULGO|CONDENO|ABSOLVO|DEFIRO|INDEFIRO|DETERMINO)\b",
    ],
    "Precedent": [
        r"\b(<CASE_REF>|<SUMULA_REF>|<ART_REF>|jurisprudência|precedente|acórdão|ementa|julgado)\b",
        r"\b(nos\s+termos\s+da|conforme\s+entendimento|segundo\s+a\s+jurisprudência|no\s+mesmo\s+sentido)\b",
        r"\b(STF|STJ|TST|TRF|TRT|TJSP|TJRJ|TJ[A-Z]{2})\b",
    ],
    "Procedural": [
        r"\b(<PROC_NUM>|petição|despacho|intimação|citação|notificação|prazo|audiência|sessão|pauta)\b",
        r"\b(intime[-\s]se|cite[-\s]se|notifique[-\s]se|cumpra[-\s]se|publique[-\s]se|registre[-\s]se)\b",
        r"\b(autos\s+remetidos|vista\s+ao\s+ministério\s+público|conclusos)\b",
    ],
}


@dataclass
class LabeledSentence:
    """
    A sentence annotated with its rhetorical role.

    Attributes:
        text: Sentence text after citation normalization
        role: Predicted rhetorical role label
        confidence: Confidence score for the predicted role
        citation_metadata: Original citations extracted from this sentence
    """

    text: str
    role: str
    confidence: float
    citation_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LabelingOutput:
    """
    Output of the rhetorical role labeling step.

    Attributes:
        labeled_sentences: List of sentences with role assignments
        role_distribution: Count per role across all sentences
        uncertain_count: Number of sentences labeled Uncertain
        source_path: Propagated from previous step
    """

    labeled_sentences: list[LabeledSentence]
    role_distribution: dict[str, int] = field(default_factory=dict)
    uncertain_count: int = 0
    source_path: Optional[object] = None


class RhetoricalLabeler(PipelineStep):
    """
    Assign rhetorical roles to legal sentences.

    This is a keyword-based heuristic classifier acting as a placeholder
    for a fine-tuned LegalBERT-pt model. It matches sentences against
    curated regex patterns per role and assigns the highest-scoring role.
    Sentences with no pattern match or confidence below the threshold are
    labeled Uncertain. The interface is designed so that replacing this
    class with a transformer-based implementation requires no downstream
    changes.
    """

    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize rhetorical labeler.

        Args:
            confidence_threshold: Minimum score to accept a role assignment
        """
        super().__init__(
            step_number=5,
            name="Rhetorical Labeler",
            description="Classify sentences into 5 rhetorical roles",
        )
        self._confidence_threshold = confidence_threshold

    def process(self, input_data: CitationOutput) -> LabelingOutput:
        """
        Label all sentences with rhetorical roles.

        Args:
            input_data: CitationOutput with sentences and citation_metadata

        Returns:
            LabelingOutput with labeled sentences and role statistics
        """
        labeled: list[LabeledSentence] = []
        for sentence, citations in zip(input_data.sentences, input_data.citation_metadata):
            role, confidence = self._classify(sentence)
            labeled.append(LabeledSentence(
                text=sentence,
                role=role,
                confidence=confidence,
                citation_metadata=citations,
            ))
        role_distribution = {role: 0 for role in ROLES}
        for item in labeled:
            role_distribution[item.role] = role_distribution.get(item.role, 0) + 1
        uncertain_count = role_distribution.get("Uncertain", 0)
        return LabelingOutput(
            labeled_sentences=labeled,
            role_distribution=role_distribution,
            uncertain_count=uncertain_count,
            source_path=input_data.source_path,
        )

    def _classify(self, sentence: str) -> tuple[str, float]:
        """
        Score sentence against all role pattern groups.

        Args:
            sentence: Normalized sentence text

        Returns:
            Tuple of (best_role, confidence_score)
        """
        scores: dict[str, float] = {}
        for role, patterns in _ROLE_PATTERNS.items():
            match_count = sum(
                1 for p in patterns if re.search(p, sentence, re.IGNORECASE)
            )
            scores[role] = match_count / len(patterns)
        best_role = max(scores, key=lambda r: scores[r])
        best_score = scores[best_role]
        if best_score < self._confidence_threshold / len(_ROLE_PATTERNS[best_role]):
            return "Uncertain", 0.0
        return best_role, round(best_score, 4)

    def validate(self, output_data: LabelingOutput) -> bool:
        """
        Validate that labeled sentences were produced.

        Args:
            output_data: LabelingOutput to validate

        Returns:
            True if labeled_sentences is non-empty
        """
        return len(output_data.labeled_sentences) > 0
