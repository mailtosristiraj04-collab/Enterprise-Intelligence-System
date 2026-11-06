"""Document format extractors used by the ingestion pipeline."""

from dataclasses import dataclass, field
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend.core.exceptions import DocumentProcessingError, ExtractionDependencyError
from backend.utils.file_utils import estimate_tokens


@dataclass(frozen=True)
class ExtractedPage:
    """Text and metadata extracted from one logical page."""

    text: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Structured output shared by every extractor."""

    pages: list[ExtractedPage]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


class BaseExtractor:
    """Interface implemented by document extractors."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        raise NotImplementedError


class PdfExtractor(BaseExtractor):
    """Extract text and metadata from PDF files with PyMuPDF."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        try:
            import fitz
        except ImportError as exc:
            raise ExtractionDependencyError("PDF", "PyMuPDF", "Install PyMuPDF.") from exc
        try:
            with fitz.open(stream=content, filetype="pdf") as document:
                pages = [
                    ExtractedPage(
                        text=page.get_text("text"),
                        page_number=index + 1,
                        metadata={"token_estimate": estimate_tokens(page.get_text("text"))},
                    )
                    for index, page in enumerate(document)
                ]
                return ExtractionResult(
                    pages=pages,
                    metadata={"page_count": len(pages), **document.metadata},
                    warnings=[
                        f"Page {page.page_number} contains no text."
                        for page in pages
                        if not page.text.strip()
                    ],
                )
        except Exception as exc:
            raise DocumentProcessingError(f"Unable to extract PDF '{filename}'.") from exc


class DocxExtractor(BaseExtractor):
    """Extract paragraphs and table content from DOCX files."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        try:
            from docx import Document
        except ImportError as exc:
            raise ExtractionDependencyError("DOCX", "python-docx", "Install python-docx.") from exc
        try:
            document = Document(io.BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            paragraphs.extend(
                cell.text
                for table in document.tables
                for row in table.rows
                for cell in row.cells
            )
            properties = document.core_properties
            metadata = {
                "author": properties.author,
                "title": properties.title,
                "subject": properties.subject,
                "paragraph_count": len(paragraphs),
            }
            return ExtractionResult(
                pages=[ExtractedPage(text="\n".join(paragraphs), metadata=metadata)],
                metadata=metadata,
            )
        except Exception as exc:
            raise DocumentProcessingError(f"Unable to extract DOCX '{filename}'.") from exc


class DocExtractor(BaseExtractor):
    """Extract legacy DOC files through a locally installed LibreOffice binary."""

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        with tempfile.TemporaryDirectory(prefix="eki-doc-") as directory:
            source = Path(directory) / "document.doc"
            source.write_bytes(content)
            try:
                completed = subprocess.run(
                    ["soffice", "--headless", "--convert-to", "txt:Text", "--outdir", directory, str(source)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
            except FileNotFoundError as exc:
                raise ExtractionDependencyError(
                    "DOC", "LibreOffice", "Install LibreOffice and ensure 'soffice' is on PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise DocumentProcessingError(f"Timed out while extracting DOC '{filename}'.") from exc
            output = Path(directory) / "document.txt"
            if completed.returncode != 0 or not output.exists():
                detail = completed.stderr.strip() or "LibreOffice could not convert the document."
                raise DocumentProcessingError(f"Unable to extract DOC '{filename}'.", detail=detail)
            return ExtractionResult(pages=[ExtractedPage(text=output.read_text(encoding="utf-8"))])


class TxtExtractor(BaseExtractor):
    """Extract UTF-8 text with a small set of common fallback encodings."""

    _encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    def extract(self, content: bytes, filename: str) -> ExtractionResult:
        for encoding in self._encodings:
            try:
                text = content.decode(encoding)
                warning = [] if encoding in ("utf-8-sig", "utf-8") else [f"Decoded using {encoding}."]
                return ExtractionResult(pages=[ExtractedPage(text=text)], warnings=warning)
            except UnicodeDecodeError:
                continue
        raise DocumentProcessingError(f"Unable to decode text file '{filename}'.")


def get_extractor(extension: str) -> BaseExtractor:
    """Return the extractor for a normalized file extension."""
    extractors: dict[str, BaseExtractor] = {
        ".pdf": PdfExtractor(),
        ".docx": DocxExtractor(),
        ".doc": DocExtractor(),
        ".txt": TxtExtractor(),
    }
    try:
        return extractors[extension.lower()]
    except KeyError as exc:
        raise DocumentProcessingError(f"No extractor is available for '{extension}'.") from exc
