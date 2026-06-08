import os
import re
from dataclasses import dataclass
from io import BytesIO

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


# These defaults worked well for the current project:
# OCR is off because the tested PDFs are digital, not scanned images.
# Table structure is on because some papers store important information in tables.
ENABLE_OCR = False
ENABLE_TABLE_STRUCTURE = True
FORCE_BACKEND_TEXT = True


@dataclass(frozen=True)
class DoclingChunk:
    index: int
    text: str


def _clean_chunk_text(text: str) -> str:
    #Normalize whitespace without changing the actual meaning of the text.
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_docling_converter(artifacts_path: str | None = None) -> DocumentConverter:
    """
    Build the Docling converter used for PDF parsing.

    The optional artifacts path is useful if Docling models are stored locally
    or configured through the DOCLING_ARTIFACTS_PATH environment variable.
    """
    if artifacts_path is None:
        artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH")

    pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    pipeline_options.do_ocr = ENABLE_OCR
    pipeline_options.do_table_structure = ENABLE_TABLE_STRUCTURE
    pipeline_options.force_backend_text = FORCE_BACKEND_TEXT

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def parse_pdf_with_docling(
    file_bytes: bytes,
    filename: str = "uploaded.pdf",
    converter: DocumentConverter | None = None,
) -> tuple[str, list[DoclingChunk]]:
    """
    Parse a PDF from uploaded file bytes and return:
    1. the full extracted text
    2. smaller chunks used later for retrieval and question answering
    """
    if converter is None:
        converter = build_docling_converter()

    source = DocumentStream(
        name=filename,
        stream=BytesIO(file_bytes),
    )

    result = converter.convert(source)
    document = result.document

    full_text = _clean_chunk_text(document.export_to_text())

    chunker = HierarchicalChunker()
    raw_chunks = list(chunker.chunk(document))

    chunks: list[DoclingChunk] = []

    for raw_chunk in raw_chunks:
        chunk_text = _clean_chunk_text(raw_chunk.text)

        if not chunk_text:
            continue

        # Use len(chunks) instead of enumerate(raw_chunks) so indexes stay
        # aligned with the final filtered chunk list.
        chunks.append(
            DoclingChunk(
                index=len(chunks),
                text=chunk_text,
            )
        )

    return full_text, chunks