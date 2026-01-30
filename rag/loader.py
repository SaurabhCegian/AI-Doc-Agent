from pathlib import Path
from typing import List

def _load_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to read PDF files. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text)

def _load_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def load_document(doc_dir: str) -> List[str]:
    """Load Document from directory"""
    texts = []
    base_path = Path(doc_dir)
    for file_path in base_path.glob("*.*"):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            texts.append(_load_pdf(file_path))
        elif suffix in {".txt", ".md"}:
            texts.append(_load_text_file(file_path))

    return texts

def chunk_text(text:str, chunk_size:int=500, chunk_overlap:int=100) -> list[str]:
    """Chunk text into smaller pieces"""
    chunks=[]
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def load_and_chunk_docs(doc_dir: str) -> List[str]:
    """Load Document and chunk them"""
    print(f"Loading documents from {doc_dir}...")
    documents = load_document(doc_dir)
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    return all_chunks