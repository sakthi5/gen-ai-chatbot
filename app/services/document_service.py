import io

from pypdf import PdfReader
from docx import Document as DocxDocument

# Simple approach: the whole extracted document is injected into the
# conversation's context (see chat_service.load_conversation_history), so we
# cap how much text we keep per document to leave room for the actual
# conversation and the model's answer. ~20k characters is roughly 5k tokens.
MAX_DOCUMENT_CHARACTERS = 20_000

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")


class UnsupportedDocumentType(ValueError):
    """Raised when a file's extension isn't one we know how to read."""


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract plain text from an uploaded .pdf, .txt, or .docx file.

    Raises UnsupportedDocumentType for anything else.
    """

    extension = _get_extension(filename)

    if extension == ".pdf":
        text = _extract_pdf_text(file_bytes)
    elif extension == ".docx":
        text = _extract_docx_text(file_bytes)
    elif extension == ".txt":
        text = _extract_txt_text(file_bytes)
    else:
        raise UnsupportedDocumentType(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = text.strip()

    if len(text) > MAX_DOCUMENT_CHARACTERS:
        text = (
            text[:MAX_DOCUMENT_CHARACTERS]
            + "\n\n[... document truncated — only the first "
            f"{MAX_DOCUMENT_CHARACTERS:,} characters were kept ...]"
        )

    return text


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx_text(file_bytes: bytes) -> str:
    document = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def _extract_txt_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")
