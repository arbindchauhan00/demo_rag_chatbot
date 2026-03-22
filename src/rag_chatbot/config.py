"""Environment-backed settings for Ollama, Chroma, and chunking."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_embed_model: str
    ollama_llm_model: str
    ollama_keep_alive_chat: str
    ollama_keep_alive_embed_seconds: int
    chroma_path: Path
    chroma_collection: str
    pdf_dir: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    max_context_chars: int


def parse_keep_alive_to_seconds(value: str) -> int:
    """Convert Ollama-style durations (e.g. 30m, 1h) to seconds for embedding API."""
    s = value.strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([mhs])?$", s)
    if not m:
        return int(float(s))
    num, unit = float(m.group(1)), m.group(2) or "s"
    mult = {"s": 1, "m": 60, "h": 3600}.get(unit, 1)
    return int(num * mult)


def load_settings() -> Settings:
    root = _project_root()
    keep = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        ollama_llm_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        ollama_keep_alive_chat=keep,
        ollama_keep_alive_embed_seconds=parse_keep_alive_to_seconds(keep),
        chroma_path=Path(os.getenv("CHROMA_PATH", str(root / "chroma_db"))),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "rag_pdf"),
        pdf_dir=Path(os.getenv("PDF_DIR", str(root / "data" / "pdfs"))),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "12000")),
    )
