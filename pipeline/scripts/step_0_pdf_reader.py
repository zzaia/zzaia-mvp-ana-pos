"""Step 0: Read text from PDF using pdfplumber (digital) and Tesseract (scanned)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

try:
    import pytesseract
    from PIL import Image as _PilImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

from pipeline_step import PipelineStep


@dataclass
class PdfReaderInput:
    """
    Input for the PDF reader step.

    Attributes:
        pdf_path: Absolute path to the PDF file
        ocr_fallback: Whether to attempt OCR when digital extraction fails
        min_text_length: Minimum character count to consider a page non-empty
    """

    pdf_path: Path
    ocr_fallback: bool = True
    min_text_length: int = 50


@dataclass
class PdfReaderOutput:
    """
    Output of the PDF reader step.

    Attributes:
        raw_text: Concatenated raw Unicode text from all pages
        page_count: Total number of pages in the PDF
        ocr_pages: Page numbers that required OCR fallback
        source_path: Resolved path of the processed PDF
    """

    raw_text: str
    page_count: int
    ocr_pages: list[int] = field(default_factory=list)
    source_path: Optional[Path] = None


class PdfReader(PipelineStep):
    """
    Extract raw Unicode text from PDF documents.

    Uses pdfplumber for digital PDFs and falls back to Tesseract
    OCR for scanned pages when the digital layer is absent.
    """

    def __init__(self, min_text_length: int = 50):
        """
        Initialize PDF reader.

        Args:
            min_text_length: Minimum characters per page to skip OCR fallback
        """
        super().__init__(
            step_number=0,
            name="PDF Reader",
            description="Extract raw Unicode text from PDF pages",
        )
        self._min_text_length = min_text_length

    def process(self, input_data: PdfReaderInput) -> PdfReaderOutput:
        """
        Extract text from all PDF pages.

        Args:
            input_data: PdfReaderInput with pdf_path and options

        Returns:
            PdfReaderOutput with concatenated raw text
        """
        pdf_path = Path(input_data.pdf_path).resolve()
        pages_text: list[str] = []
        ocr_pages: list[int] = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < input_data.min_text_length and input_data.ocr_fallback:
                    text = self._ocr_page(page, idx + 1)
                    ocr_pages.append(idx + 1)
                pages_text.append(text)
        raw_text = "\n\n".join(pages_text)
        return PdfReaderOutput(
            raw_text=raw_text,
            page_count=page_count,
            ocr_pages=ocr_pages,
            source_path=pdf_path,
        )

    def _ocr_page(self, page: object, page_num: int) -> str:
        """
        Apply Tesseract OCR to a single scanned page.

        Args:
            page: pdfplumber page object
            page_num: 1-based page number for logging

        Returns:
            Extracted text string from OCR
        """
        if not _OCR_AVAILABLE:
            logging.getLogger("pipeline.step_0").warning(
                f"pytesseract/Pillow not available; page {page_num} returned empty"
            )
            return ""
        image: _PilImage.Image = page.to_image(resolution=300).original
        text: str = pytesseract.image_to_string(image, lang="por")
        return text

    def validate(self, output_data: PdfReaderOutput) -> bool:
        """
        Validate that text was extracted.

        Args:
            output_data: PdfReaderOutput to validate

        Returns:
            True if raw_text is non-empty
        """
        return bool(output_data.raw_text.strip())
