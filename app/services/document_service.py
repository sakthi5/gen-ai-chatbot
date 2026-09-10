import io

from pypdf import PdfReader
from docx import Document as DocxDocument

# Simple approach: the whole extracted document is injected into the
# conversation's context on EVERY message (see
# chat_service.load_conversation_history), not just once — so its size
# eats into the same per-minute token budget repeatedly, not a one-time
# cost. Groq's free/on-demand tier caps openai/gpt-oss-20b at 8,000
# tokens/minute total (prompt + expected output); a single ~16k-character
# document alone (~4k tokens) left too little room for the system prompt,
# conversation history, and the model's answer, and reliably 413'd
# ("Request too large ... on tokens per minute"). 8,000 characters is
# roughly 2k tokens — leaves real headroom for everything else.
MAX_DOCUMENT_CHARACTERS = 8_000

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
