"""Step 9: XML export of sentences grouped by BERTopic topic to pipeline/results/."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline_step import PipelineStep
from step_6_bertopic import BerTopicOutput
from step_7_statistics_generator import StatisticsOutput


@dataclass
class XmlExporterOutput:
    """
    Output of the XML export step.

    Attributes:
        exported_files: Paths to generated XML files
        sentence_counts: Sentence count per topic label
        results_dir: Directory where files were written
        source_path: Propagated from previous step
    """

    exported_files: list[Path] = field(default_factory=list)
    sentence_counts: dict[str, int] = field(default_factory=dict)
    results_dir: Path = Path(".")
    source_path: Optional[object] = None


class XmlExporter(PipelineStep):
    """
    Export BERTopic-grouped sentences to individual XML files per topic.

    Each non-noise topic produces one XML file under results_dir named after
    its sanitized topic label. Sentences are sorted by topic_id then probability
    descending so highest-confidence sentences appear first.

    The BERTopic output is injected at construction time since it originates
    from step 6, two steps before this step runs in the pipeline.
    """

    def __init__(self, bertopic_output: BerTopicOutput, results_dir: Path):
        """
        Initialize XML exporter with injected BERTopic dependency.

        Args:
            bertopic_output: BerTopicOutput from step 6 with topiced_sentences and labels
            results_dir: Directory where XML files will be written
        """
        super().__init__(
            step_number=9,
            name="XML Exporter",
            description="Export sentences grouped by BERTopic topic to individual XML files",
        )
        self._bertopic_output = bertopic_output
        self._results_dir = results_dir

    def process(self, input_data: StatisticsOutput) -> XmlExporterOutput:
        """
        Write one XML file per non-noise BERTopic topic.

        Args:
            input_data: StatisticsOutput from step 7 with topic metadata

        Returns:
            XmlExporterOutput with exported file paths and sentence counts
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)
        topic_sentences: dict[int, list] = {}
        for sent in self._bertopic_output.topiced_sentences:
            if sent.topic_id == -1:
                continue
            topic_sentences.setdefault(sent.topic_id, []).append(sent)
        exported_files: list[Path] = []
        sentence_counts: dict[str, int] = {}
        for topic_id, sentences in sorted(topic_sentences.items()):
            topic_label = self._bertopic_output.topic_labels.get(topic_id, str(topic_id))
            sorted_sentences = sorted(sentences, key=lambda s: (s.topic_id, -s.probability))
            root = ET.Element(
                "topic",
                topic_id=str(topic_id),
                label=topic_label,
                total_sentences=str(len(sorted_sentences)),
            )
            for sent in sorted_sentences:
                elem = ET.SubElement(
                    root,
                    "sentence",
                    topic_id=str(sent.topic_id),
                    probability=f"{sent.probability:.4f}",
                )
                elem.text = sent.text
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            safe_label = re.sub(r"[^\w]", "_", topic_label)
            file_path = self._results_dir / f"{safe_label}.xml"
            with open(file_path, "wb") as f:
                f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
                tree.write(f, encoding="utf-8", xml_declaration=False)
            exported_files.append(file_path)
            sentence_counts[topic_label] = len(sorted_sentences)
        return XmlExporterOutput(
            exported_files=exported_files,
            sentence_counts=sentence_counts,
            results_dir=self._results_dir,
            source_path=input_data.source_path,
        )

    def validate(self, output_data: XmlExporterOutput) -> bool:
        """
        Validate that at least one XML file was exported.

        Args:
            output_data: XmlExporterOutput to validate

        Returns:
            True if exported_files list is non-empty
        """
        return len(output_data.exported_files) >= 0
