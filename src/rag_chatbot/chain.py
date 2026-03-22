"""RAG chain: Chroma retriever + ChatOllama with async streaming."""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from rag_chatbot.config import Settings

log = logger.bind(scope="chain")

SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions using only the provided context from PDF documents.
If the answer is not in the context, say that you do not know based on the documents.
Be concise and accurate.

Context:
{context}"""


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def get_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive_embed_seconds,
    )


def load_vectorstore(settings: Settings) -> Chroma:
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(settings),
        collection_name=settings.chroma_collection,
    )


def build_rag_components(
    settings: Settings,
) -> tuple[Runnable[Any, Any], Runnable[Any, Any], Runnable[Any, Any]]:
    """Return ``(retrieve_runnable, generate_runnable, full_chain)`` for step-wise UI and logging."""
    store = load_vectorstore(settings)
    retriever = store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    llm = ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive_chat,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "{input}"),
        ]
    )

    async def retrieve_and_pack(b: dict) -> dict:
        q = b["input"]
        log.info("Embedding query and retrieving up to {} chunks from Chroma", settings.retrieval_k)
        docs = await retriever.ainvoke(q)
        log.debug("Retriever returned {} chunks", len(docs))
        body = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted({str(doc.metadata.get("source", "?")) for doc in docs})
        footer = f"\n\n[Document sources: {', '.join(sources)}]" if sources else ""
        packed = _truncate_context(body + footer, settings.max_context_chars)
        log.info(
            "Context packed: {} chars (truncated to max {})",
            len(packed),
            settings.max_context_chars,
        )
        return {
            "input": q,
            "context": packed,
            "sources": sources,
            "retrieved_chunk_count": len(docs),
        }

    retrieve_runnable = RunnableLambda(retrieve_and_pack)
    generate_runnable = prompt | llm
    full_chain = retrieve_runnable | generate_runnable
    return retrieve_runnable, generate_runnable, full_chain


def build_rag_chain(settings: Settings) -> Runnable[Any, Any]:
    """Single runnable: retrieve → prompt → LLM."""
    _, _, full_chain = build_rag_components(settings)
    return full_chain


def _token_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return ""


async def stream_answer_tokens(rag_chain: Runnable[Any, Any], question: str) -> AsyncIterator[str]:
    """Yield streamed text tokens from the chat model (retrieval runs first)."""
    async for event in rag_chain.astream_events(
        {"input": question},
        version="v2",
    ):
        if event.get("event") != "on_chat_model_stream":
            continue
        data = event.get("data") or {}
        chunk = data.get("chunk")
        if chunk is None:
            continue
        text = _token_from_chunk(chunk)
        if text:
            yield text


async def stream_generation_tokens(
    generate_runnable: Runnable[Any, Any],
    packed: dict,
) -> AsyncIterator[str]:
    """Stream tokens from ``prompt | llm`` given packed context (after retrieval)."""
    log.info("Starting LLM token stream (prompt | ChatOllama)")
    async for event in generate_runnable.astream_events(packed, version="v2"):
        if event.get("event") != "on_chat_model_stream":
            continue
        data = event.get("data") or {}
        chunk = data.get("chunk")
        if chunk is None:
            continue
        text = _token_from_chunk(chunk)
        if text:
            yield text


async def ainvoke_answer(rag_chain: Runnable[Any, Any], question: str) -> Any:
    """Non-streaming full AIMessage response."""
    return await rag_chain.ainvoke({"input": question})
