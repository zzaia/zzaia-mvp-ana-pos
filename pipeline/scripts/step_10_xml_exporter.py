"""Step 10: XML export of sentences grouped by case type to pipeline/results/."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline_step import PipelineStep
from step_7_hdbscan_clusterer import ClusteringOutput
from step_8_statistics_generator import StatisticsOutput


@dataclass
class XmlExportOutput:
    """
    Output of the XML export step.

    Attributes:
        exported_files: Paths to generated XML files
        sentence_counts: Sentence count per case type
        results_dir: Directory where files were written
        source_path: Propagated from previous step
    """

    exported_files: list[Path] = field(default_factory=list)
    sentence_counts: dict[str, int] = field(default_factory=dict)
    results_dir: Path = Path(".")
    source_path: Optional[object] = None


class XmlExporter(PipelineStep):
    """
    Export clustered sentences grouped by case type to individual XML files.

    Each non-Unknown case type produces one XML file under results_dir.
    Sentences are sourced from the injected ClusteringOutput and sorted by
    cluster_id then membership_probability descending so the highest-confidence
    sentences appear first within each cluster group.

    The clustering output is injected at construction time since it originates
    from step 7, two steps before this step runs in the pipeline.
    """

    def __init__(self, clustering_output: ClusteringOutput, results_dir: Path):
        """
        Initialize XML exporter with injected clustering dependency.

        Args:
            clustering_output: ClusteringOutput from step 7 with clustered sentences
            results_dir: Directory where XML files will be written
        """
        super().__init__(
            step_number=10,
            name="XML Exporter",
            description="Export sentences grouped by case type to individual XML files",
        )
        self._clustering_output = clustering_output
        self._results_dir = results_dir

    def process(self, input_data: StatisticsOutput) -> XmlExportOutput:
        """
        Write one XML file per non-Unknown case type.

        Args:
            input_data: StatisticsOutput from step 8 with cluster_stats for case type mapping

        Returns:
            XmlExportOutput with exported file paths and sentence counts
        """
        self._results_dir.mkdir(parents=True, exist_ok=True)
        cluster_case_map: dict[int, str] = {
            s.cluster_id: s.case_type for s in input_data.cluster_stats
        }
        case_type_sentences: dict[str, list] = {}
        for sent in self._clustering_output.clustered_sentences:
            if sent.cluster_id == -1:
                continue
            case_type = cluster_case_map.get(sent.cluster_id, "Unknown")
            if case_type == "Unknown":
                continue
            case_type_sentences.setdefault(case_type, []).append(sent)
        exported_files: list[Path] = []
        sentence_counts: dict[str, int] = {}
        for case_type, sentences in sorted(case_type_sentences.items()):
            sorted_sentences = sorted(sentences, key=lambda s: (s.cluster_id, -s.membership_probability))
            root = ET.Element("case_type", name=case_type, total_sentences=str(len(sorted_sentences)))
            for sent in sorted_sentences:
                elem = ET.SubElement(
                    root,
                    "sentence",
                    cluster_id=str(sent.cluster_id),
                    membership_probability=f"{sent.membership_probability:.4f}",
                )
                elem.text = sent.text
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            file_path = self._results_dir / f"{case_type}.xml"
            with open(file_path, "wb") as f:
                f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
                tree.write(f, encoding="utf-8", xml_declaration=False)
            exported_files.append(file_path)
            sentence_counts[case_type] = len(sorted_sentences)
        return XmlExportOutput(
            exported_files=exported_files,
            sentence_counts=sentence_counts,
            results_dir=self._results_dir,
            source_path=input_data.source_path,
        )

    def validate(self, output_data: XmlExportOutput) -> bool:
        """
        Validate that at least one XML file was exported.

        Args:
            output_data: XmlExportOutput to validate

        Returns:
            True if exported_files list is non-empty
        """
        return len(output_data.exported_files) > 0
