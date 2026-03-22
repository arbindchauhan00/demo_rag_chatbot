"""Load PDFs, chunk, embed with Ollama, persist to Chroma."""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from rag_chatbot.config import Settings

log = logger.bind(scope="ingest")


def _list_pdfs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.pdf"))


def run_ingest(settings: Settings, pdf_dir: Path | None = None, *, clear_collection: bool = True) -> int:
    """Ingest all PDFs from ``pdf_dir`` into Chroma. Returns number of chunks stored."""
    root = Path(pdf_dir) if pdf_dir is not None else settings.pdf_dir
    log.info("PDF directory: {}", root.resolve())
    pdfs = _list_pdfs(root)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {root}")
    log.info("Found {} PDF file(s): {}", len(pdfs), [p.name for p in pdfs])

    documents = []
    for path in pdfs:
        log.debug("Loading {}", path.name)
        loader = PyPDFLoader(str(path))
        for doc in loader.load():
            doc.metadata.setdefault("source", path.name)
            documents.append(doc)
    log.info("Loaded {} raw document page(s) from PDFs", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    splits = splitter.split_documents(documents)
    log.info(
        "Split into {} chunks (chunk_size={}, overlap={})",
        len(splits),
        settings.chunk_size,
        settings.chunk_overlap,
    )

    log.info(
        "Using Ollama embeddings model={} at {}",
        settings.ollama_embed_model,
        settings.ollama_base_url,
    )
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive_embed_seconds,
    )

    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    if clear_collection:
        log.info("Clearing existing collection '{}' if present", settings.chroma_collection)
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        existing = {c.name for c in client.list_collections()}
        if settings.chroma_collection in existing:
            client.delete_collection(settings.chroma_collection)
            log.debug("Deleted collection {}", settings.chroma_collection)

    log.info(
        "Writing vectors to Chroma persist_directory={} collection={}",
        settings.chroma_path,
        settings.chroma_collection,
    )
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(settings.chroma_path),
        collection_name=settings.chroma_collection,
    )
    log.info("Ingest finished: {} chunks indexed", len(splits))
    return len(splits)
