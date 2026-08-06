import os
import time
from typing import List, Dict, Any, Optional
from config.settings import settings
from config.logging_config import logger
from src.metadata.schema import Document
from src.ingestion.txt_loader import TextDocumentLoader
from src.ingestion.pdf_loader import PDFDocumentLoader
from src.ingestion.docx_loader import DocxDocumentLoader
from src.ingestion.doc_loader import DocDocumentLoader
from src.ingestion.spreadsheet_loader import SpreadsheetDocumentLoader
from src.ingestion.html_loader import HTMLDocumentLoader
from src.ingestion.web_loader import WebPageLoader
from src.ingestion.fingerprint import (
    compute_source_fingerprint,
    file_updated_at_iso,
    normalize_source_path,
    stable_document_id,
)

from src.chunking.recursive import RecursiveCharacterChunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.header_aware import HeaderAwareChunker

from src.embeddings.mock import MockEmbeddingProvider
from src.embeddings.ollama import OllamaEmbeddingProvider
from src.embeddings.openai import OpenAIEmbeddingProvider
from src.embeddings.sentence_tf import SentenceTransformerEmbeddingProvider

from src.vector_store.memory_store import MemoryVectorStore
from src.vector_store.qdrant_store import QdrantVectorStore
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.faiss_store import FAISSVectorStore

from src.retrieval.vector_search import VectorSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.hybrid import HybridSearchEngine
from src.retrieval.query_rewriter import QueryRewriter
from src.reranking.identity import IdentityReranker
from src.reranking.cross_encoder import CrossEncoderReranker

from src.context.builder import ContextBuilder
from src.context.citation import CitationFormatter
from src.context.answer_formatter import AnswerFormatter
from src.context.structured_prompt import augment_system_prompt
from src.context.response_length import resolve_max_output_tokens, resolve_context_tokens, parse_requested_words, augment_chat_system_prompt
from src.context.conversation import build_rag_user_message, resolve_retrieval_query
from src.llm.base import SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from src.cache.query_cache import QueryCache

from src.llm.mock import MockLLMProvider
from src.llm.ollama import OllamaLLMProvider
from src.llm.nvidia_nim import NvidiaNIMLLMProvider
from src.llm.openai import OpenAILLMProvider
from src.llm.gemini import GeminiLLMProvider

from src.evaluation.metrics import EvaluationMetrics
from src.security.validator import InputValidator
from src.monitoring.metrics import metrics_collector
from src.monitoring.logger import QueryTelemetryLogger


class RAGPipelineService:
    """Central orchestrator for the production RAG architecture."""

    def __init__(self):
        # 1. Initialize Embedding Provider
        if settings.EMBEDDING_PROVIDER == "ollama":
            self.embedding_provider = OllamaEmbeddingProvider()
        elif settings.EMBEDDING_PROVIDER == "openai":
            self.embedding_provider = OpenAIEmbeddingProvider()
        elif settings.EMBEDDING_PROVIDER == "sentence_transformers":
            self.embedding_provider = SentenceTransformerEmbeddingProvider()
        elif settings.EMBEDDING_PROVIDER == "mock":
            self.embedding_provider = MockEmbeddingProvider()
        else:
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")

        # 2. Initialize Vector Store
        if settings.VECTOR_STORE_PROVIDER == "qdrant":
            self.vector_store = QdrantVectorStore()
        elif settings.VECTOR_STORE_PROVIDER == "chroma":
            self.vector_store = ChromaVectorStore(settings.VECTOR_DB_PATH)
        elif settings.VECTOR_STORE_PROVIDER == "faiss":
            self.vector_store = FAISSVectorStore()
        else:
            self.vector_store = ChromaVectorStore(persist_directory=settings.VECTOR_DB_PATH)

        # 3. Initialize Retrieval Engines
        self.vector_engine = VectorSearchEngine(self.vector_store, self.embedding_provider)
        self.bm25_engine = BM25SearchEngine(k1=settings.BM25_K1, b=settings.BM25_B)
        self.hybrid_engine = HybridSearchEngine(
            vector_engine=self.vector_engine,
            bm25_engine=self.bm25_engine,
            alpha=settings.HYBRID_ALPHA,
            rrf_k=settings.RRF_K
        )

        # 4. Initialize Reranker
        if settings.RERANKER_PROVIDER == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = IdentityReranker()

        # 5. Initialize Context Builder
        self.context_builder = ContextBuilder(max_tokens=settings.DEFAULT_CONTEXT_TOKENS)

        # 6. Initialize LLM Provider
        if settings.LLM_PROVIDER == "ollama":
            self.llm_provider = OllamaLLMProvider()
        elif settings.LLM_PROVIDER == "nvidia_nim":
            self.llm_provider = NvidiaNIMLLMProvider()
        elif settings.LLM_PROVIDER == "openai":
            self.llm_provider = OpenAILLMProvider()
        elif settings.LLM_PROVIDER == "gemini":
            self.llm_provider = GeminiLLMProvider()
        elif settings.LLM_PROVIDER == "mock":
            self.llm_provider = MockLLMProvider()
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")

        # 7. Query rewriter for conversational follow-ups
        self.query_rewriter = QueryRewriter(
            self.llm_provider,
            max_history_turns=settings.MAX_CHAT_HISTORY_TURNS,
        )

        self._all_chunk_documents: List[Document] = []
        self._query_cache = QueryCache()
        self._rehydrate_bm25_index()

    def get_llm_model_name(self) -> str:
        if hasattr(self.llm_provider, "model"):
            return str(self.llm_provider.model)
        return settings.OLLAMA_LLM_MODEL if settings.LLM_PROVIDER == "ollama" else settings.LLM_PROVIDER

    def set_llm_model(self, model: str) -> None:
        if hasattr(self.llm_provider, "set_model"):
            self.llm_provider.set_model(model)
            self._query_cache.clear()
            return
        raise ValueError(f"Model switching is not supported for provider '{settings.LLM_PROVIDER}'.")

    def apply_request_model(self, model: Optional[str]) -> None:
        """Switch active LLM when the client sends a per-request model override."""
        if not model or not str(model).strip():
            return
        if settings.LLM_PROVIDER != "ollama":
            return
        requested = str(model).strip()
        if self.get_llm_model_name() != requested:
            self.set_llm_model(requested)

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        if hasattr(self.vector_store, "list_indexed_sources"):
            return self.vector_store.list_indexed_sources()
        return []

    @staticmethod
    def _grounding_flags(answer: str, match_percent: float, mode: str) -> Dict[str, bool]:
        not_found = "could not find that information" in (answer or "").lower()
        if mode == "direct":
            return {"grounded": False, "not_in_documents": True}
        grounded = match_percent >= 40 and not not_found
        return {"grounded": grounded, "not_in_documents": not grounded}

    @staticmethod
    def _retrieval_limits(query: str, top_k: int, top_m_rerank: int) -> tuple[int, int]:
        words = parse_requested_words(query)
        if words and words >= 2000:
            return max(top_k, 20), max(top_m_rerank, 12)
        if words and words >= 1000:
            return max(top_k, 15), max(top_m_rerank, 8)
        return top_k, top_m_rerank

    def _rehydrate_bm25_index(self) -> None:
        """Rebuild in-memory BM25 index from persisted vector store on startup."""
        if not hasattr(self.vector_store, "get_all_documents"):
            return

        persisted_docs = self.vector_store.get_all_documents()
        if not persisted_docs:
            return

        self._all_chunk_documents = persisted_docs
        self.bm25_engine.index(self._all_chunk_documents)
        logger.info(f"Rehydrated BM25 index with {len(persisted_docs)} chunks from vector store.")

    def _get_loader(self, file_path_or_url: str):
        if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
            return WebPageLoader()
        ext = os.path.splitext(file_path_or_url)[1].lower()
        if ext in [".pdf"]:
            return PDFDocumentLoader()
        elif ext in [".docx"]:
            return DocxDocumentLoader()
        elif ext in [".doc"]:
            return DocDocumentLoader()
        elif ext in [".csv", ".xlsx", ".xls"]:
            return SpreadsheetDocumentLoader()
        elif ext in [".html", ".htm"]:
            return HTMLDocumentLoader()
        else:
            return TextDocumentLoader()

    def _get_chunker(self, strategy: str = settings.CHUNKING_STRATEGY, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        if strategy == "fixed":
            return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
        elif strategy == "semantic":
            return SemanticChunker(chunk_size=size, chunk_overlap=overlap)
        elif strategy == "header":
            return HeaderAwareChunker(chunk_size=size, chunk_overlap=overlap)
        else:
            return RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap)

    def _get_ingest_record(self, source: str) -> Optional[Dict[str, Any]]:
        if hasattr(self.vector_store, "get_ingest_record_by_source"):
            return self.vector_store.get_ingest_record_by_source(source)
        return None

    def _source_is_unchanged(
        self,
        existing: Dict[str, Any],
        fingerprint: Optional[str],
        chunking_strategy: str,
        file_mtime_iso: Optional[str],
    ) -> bool:
        if existing.get("chunking_strategy") and existing["chunking_strategy"] != chunking_strategy:
            return False
        if fingerprint and existing.get("ingest_fingerprint"):
            return existing["ingest_fingerprint"] == fingerprint
        if file_mtime_iso and existing.get("updated_at"):
            return existing["updated_at"] == file_mtime_iso
        return False

    def _remove_source_from_bm25(self, source: str) -> None:
        normalized = normalize_source_path(source)
        self._all_chunk_documents = [
            doc
            for doc in self._all_chunk_documents
            if normalize_source_path(doc.metadata.source) != normalized
            and (doc.metadata.source_path or "") != normalized
        ]
        self.bm25_engine.index(self._all_chunk_documents)

    def _delete_source_from_store(self, source: str) -> int:
        if hasattr(self.vector_store, "delete_by_source"):
            return self.vector_store.delete_by_source(source)
        existing = self._get_ingest_record(source)
        if existing and existing.get("document_id"):
            return self.vector_store.delete([existing["document_id"]])
        return 0

    def ingest_source(
        self,
        source: str,
        document_id: Optional[str] = None,
        chunking_strategy: str = settings.CHUNKING_STRATEGY,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        allowed_roles: Optional[List[str]] = None,
        skip_if_unchanged: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Ingests a file or URL into the RAG system."""
        t0 = time.time()
        normalized_source = normalize_source_path(source)
        resolved_document_id = document_id or stable_document_id(source)
        fingerprint = compute_source_fingerprint(source)
        file_mtime_iso = file_updated_at_iso(source)

        existing = self._get_ingest_record(source)
        if existing and skip_if_unchanged and not force:
            if self._source_is_unchanged(existing, fingerprint, chunking_strategy, file_mtime_iso):
                elapsed = round(time.time() - t0, 3)
                logger.info(
                    f"Skipped unchanged source '{source}' ({existing.get('chunk_count', 0)} chunks already indexed)."
                )
                return {
                    "source": source,
                    "document_id": existing.get("document_id") or resolved_document_id,
                    "total_chunks": existing.get("chunk_count", 0),
                    "chunking_strategy": chunking_strategy,
                    "elapsed_seconds": elapsed,
                    "status": "skipped",
                    "skipped": True,
                }
            deleted = self._delete_source_from_store(source)
            if deleted:
                self._remove_source_from_bm25(source)
                logger.info(f"Removed {deleted} stale chunk(s) for updated source '{source}'.")
        elif existing and force:
            deleted = self._delete_source_from_store(source)
            if deleted:
                self._remove_source_from_bm25(source)
                logger.info(f"Removed {deleted} chunk(s) before force re-ingest of '{source}'.")

        loader = self._get_loader(source)
        raw_docs = loader.process(source=source, document_id=resolved_document_id)

        if allowed_roles:
            for doc in raw_docs:
                doc.metadata.allowed_roles = allowed_roles

        chunker = self._get_chunker(strategy=chunking_strategy, size=chunk_size, overlap=chunk_overlap)
        chunked_docs = chunker.chunk(raw_docs)

        if not chunked_docs:
            raise ValueError(
                f"No indexable chunks produced from '{source}'. "
                "The document may be empty or unsuitable for the selected chunking strategy."
            )

        for doc in chunked_docs:
            doc.metadata.source_path = normalized_source
            doc.metadata.ingest_fingerprint = fingerprint
            doc.metadata.chunking_strategy = chunking_strategy
            if file_mtime_iso:
                doc.metadata.updated_at = file_mtime_iso

        # Generate embeddings
        texts = [doc.content for doc in chunked_docs]
        embeddings = self.embedding_provider.embed_documents(texts)

        if len(embeddings) != len(chunked_docs):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(chunked_docs)} chunks."
            )
        if any(not emb for emb in embeddings):
            raise RuntimeError(
                "One or more chunks returned empty embeddings. "
                "Check that Ollama is running and nomic-embed-text is available."
            )

        # Store in vector database
        self.vector_store.add_documents(chunked_docs, embeddings)

        # Re-index BM25 search corpus
        self._all_chunk_documents.extend(chunked_docs)
        self.bm25_engine.index(self._all_chunk_documents)

        elapsed = round(time.time() - t0, 3)
        metrics_collector.record_ingestion(len(chunked_docs))

        status = "replaced" if existing else "ingested"
        logger.info(f"Ingested '{source}': {len(chunked_docs)} chunks created in {elapsed}s.")
        return {
            "source": source,
            "document_id": raw_docs[0].metadata.document_id if raw_docs else resolved_document_id,
            "total_chunks": len(chunked_docs),
            "chunking_strategy": chunking_strategy,
            "elapsed_seconds": elapsed,
            "status": status,
            "skipped": False,
        }

    def query(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        top_m_rerank: int = settings.TOP_K_RERANK,
        user_roles: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        use_cache: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes RAG or direct LLM query pipeline."""
        self.apply_request_model(model)
        model_name = self.get_llm_model_name()
        if use_cache:
            cached = self._query_cache.get(query, use_rag, model_name, chat_history)
            if cached:
                return cached

        if not use_rag:
            result = self._query_direct(query=query, chat_history=chat_history, user_roles=user_roles)
        else:
            result = self._query_rag(
                query=query,
                top_k=top_k,
                top_m_rerank=top_m_rerank,
                user_roles=user_roles,
                filter_metadata=filter_metadata,
                chat_history=chat_history,
            )

        if use_cache and not result.get("cached"):
            self._query_cache.set(query, use_rag, model_name, chat_history, result)
        return result

    def _query_rag(
        self,
        query: str,
        top_k: int,
        top_m_rerank: int,
        user_roles: Optional[List[str]],
        filter_metadata: Optional[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)

        t_rewrite_0 = time.time()
        retrieval_query = resolve_retrieval_query(
            clean_query,
            chat_history,
            self.query_rewriter,
            settings.ENABLE_QUERY_REWRITING,
        )
        rewrite_latency = round(time.time() - t_rewrite_0, 4)

        top_k, top_m_rerank = self._retrieval_limits(clean_query, top_k, top_m_rerank)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        context_tokens = resolve_context_tokens(clean_query)

        t_ret_0 = time.time()
        retrieved = self.hybrid_engine.search(
            query=retrieval_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            allowed_roles=roles,
            use_rrf=True,
        )
        ret_latency = round(time.time() - t_ret_0, 4)

        t_rank_0 = time.time()
        reranked = self.reranker.rerank(query=retrieval_query, documents=retrieved, top_n=top_m_rerank)
        rank_latency = round(time.time() - t_rank_0, 4)

        context_str, citations = self.context_builder.build_context(reranked, max_tokens=context_tokens)
        system_prompt = augment_system_prompt(SYSTEM_PROMPT, clean_query)

        t_gen_0 = time.time()
        generation_res = self.llm_provider.generate(
            prompt=clean_query,
            context=context_str,
            system_prompt=system_prompt,
            max_tokens=max_output_tokens,
            chat_history=chat_history,
        )
        gen_latency = round(time.time() - t_gen_0, 4)
        total_latency = round(time.time() - t0, 4)

        citations = AnswerFormatter.normalize_citations(citations)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])
        formatted_citations = CitationFormatter.format_citations(citations)
        match_percent = round(EvaluationMetrics.faithfulness(clean_answer, context_str) * 100, 1)
        grounding = self._grounding_flags(clean_answer, match_percent, "rag")

        metrics_collector.record_query(total_latency, generation_res.get("completion_tokens", 0), success=True)
        QueryTelemetryLogger.log_query_execution(
            query=clean_query,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            latency_breakdown={
                "rewrite": rewrite_latency,
                "retrieval": ret_latency,
                "rerank": rank_latency,
                "generation": gen_latency,
                "total": total_latency,
            },
            token_usage={
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            model=generation_res.get("model", "unknown"),
            user_roles=roles,
        )

        return {
            "query": clean_query,
            "rewritten_query": retrieval_query if retrieval_query != clean_query else None,
            "answer": clean_answer,
            "citations": citations,
            "formatted_citations": formatted_citations,
            "retrieved_context": context_str,
            "model": generation_res.get("model"),
            "token_usage": {
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            "latency": {
                "rewrite_seconds": rewrite_latency,
                "retrieval_seconds": ret_latency,
                "rerank_seconds": rank_latency,
                "generation_seconds": gen_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": len(retrieved),
            "reranked_chunks_count": len(reranked),
            "match_percent": match_percent,
            "mode": "rag",
            "cached": False,
            **grounding,
        }

    def query_stream(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        top_m_rerank: int = settings.TOP_K_RERANK,
        user_roles: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        use_cache: bool = True,
        model: Optional[str] = None,
    ):
        """Yield SSE-friendly event dicts for staged streaming responses."""
        self.apply_request_model(model)
        model_name = self.get_llm_model_name()
        if use_cache:
            cached = self._query_cache.get(query, use_rag, model_name, chat_history)
            if cached:
                yield {"type": "stage", "stage": "cached", "message": "Returning cached answer..."}
                yield {"type": "token", "content": cached["answer"]}
                yield {"type": "done", "data": cached}
                return

        if not use_rag:
            yield from self._stream_direct(query, chat_history, user_roles, use_cache, model_name)
            return

        yield from self._stream_rag(
            query, top_k, top_m_rerank, user_roles, filter_metadata, chat_history, use_cache, model_name
        )

    def _stream_direct(self, query, chat_history, user_roles, use_cache, model_name):
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        chat_system_prompt = augment_chat_system_prompt(CHAT_SYSTEM_PROMPT, clean_query)
        yield {"type": "stage", "stage": "generating", "message": "Generating answer..."}

        answer_parts: List[str] = []
        stream_fn = getattr(self.llm_provider, "chat_stream", None)
        if stream_fn:
            for token in stream_fn(
                prompt=clean_query,
                chat_history=chat_history,
                system_prompt=chat_system_prompt,
                max_tokens=max_output_tokens,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts).strip()
            model = f"mock-chat-llm" if settings.LLM_PROVIDER == "mock" else getattr(self.llm_provider, "model", "unknown")
            if settings.LLM_PROVIDER == "ollama":
                model = f"ollama/{self.llm_provider.model}"
            generation_res = {"answer": answer, "model": model, "prompt_tokens": 0, "completion_tokens": len(answer.split())}
        else:
            generation_res = self.llm_provider.chat(
                prompt=clean_query,
                chat_history=chat_history,
                system_prompt=chat_system_prompt,
                max_tokens=max_output_tokens,
            )
            answer = generation_res["answer"].strip()
            yield {"type": "token", "content": answer}

        total_latency = round(time.time() - t0, 4)
        result = {
            "query": clean_query,
            "rewritten_query": None,
            "answer": answer,
            "citations": [],
            "formatted_citations": "",
            "model": generation_res.get("model"),
            "token_usage": {
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            "latency": {
                "rewrite_seconds": 0.0,
                "retrieval_seconds": 0.0,
                "rerank_seconds": 0.0,
                "generation_seconds": total_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": 0,
            "reranked_chunks_count": 0,
            "match_percent": 0.0,
            "mode": "direct",
            "cached": False,
            **self._grounding_flags(answer, 0.0, "direct"),
        }
        if use_cache:
            self._query_cache.set(query, False, model_name, chat_history, result)
        yield {"type": "done", "data": result}

    def _stream_rag(self, query, top_k, top_m_rerank, user_roles, filter_metadata, chat_history, use_cache, model_name):
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)

        yield {"type": "stage", "stage": "rewriting", "message": "Rewriting question..."}
        retrieval_query = resolve_retrieval_query(
            clean_query,
            chat_history,
            self.query_rewriter,
            settings.ENABLE_QUERY_REWRITING,
        )

        top_k, top_m_rerank = self._retrieval_limits(clean_query, top_k, top_m_rerank)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        context_tokens = resolve_context_tokens(clean_query)

        yield {"type": "stage", "stage": "searching", "message": "Searching documents..."}
        retrieved = self.hybrid_engine.search(
            query=retrieval_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            allowed_roles=roles,
            use_rrf=True,
        )

        yield {"type": "stage", "stage": "reading", "message": f"Reading {len(retrieved)} documents..."}
        reranked = self.reranker.rerank(query=retrieval_query, documents=retrieved, top_n=top_m_rerank)
        context_str, citations = self.context_builder.build_context(reranked, max_tokens=context_tokens)
        system_prompt = augment_system_prompt(SYSTEM_PROMPT, clean_query)

        yield {"type": "stage", "stage": "generating", "message": "Generating answer..."}
        answer_parts: List[str] = []
        stream_fn = getattr(self.llm_provider, "generate_stream", None)
        if stream_fn:
            for token in stream_fn(
                prompt=clean_query,
                context=context_str,
                system_prompt=system_prompt,
                max_tokens=max_output_tokens,
                chat_history=chat_history,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts).strip()
            model = f"ollama/{self.llm_provider.model}" if settings.LLM_PROVIDER == "ollama" else "unknown"
            generation_res = {"answer": answer, "model": model, "prompt_tokens": 0, "completion_tokens": len(answer.split())}
        else:
            generation_res = self.llm_provider.generate(
                prompt=clean_query,
                context=context_str,
                system_prompt=system_prompt,
                max_tokens=max_output_tokens,
                chat_history=chat_history,
            )
            answer = generation_res["answer"]
            yield {"type": "token", "content": answer}

        total_latency = round(time.time() - t0, 4)
        citations = AnswerFormatter.normalize_citations(citations)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])
        match_percent = round(EvaluationMetrics.faithfulness(clean_answer, context_str) * 100, 1)
        result = {
            "query": clean_query,
            "rewritten_query": retrieval_query if retrieval_query != clean_query else None,
            "answer": clean_answer,
            "citations": citations,
            "formatted_citations": CitationFormatter.format_citations(citations),
            "retrieved_context": context_str,
            "model": generation_res.get("model"),
            "token_usage": {
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            "latency": {
                "rewrite_seconds": 0.0,
                "retrieval_seconds": 0.0,
                "rerank_seconds": 0.0,
                "generation_seconds": total_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": len(retrieved),
            "reranked_chunks_count": len(reranked),
            "match_percent": match_percent,
            "mode": "rag",
            "cached": False,
            **self._grounding_flags(clean_answer, match_percent, "rag"),
        }
        if use_cache:
            self._query_cache.set(query, True, model_name, chat_history, result)
        yield {"type": "done", "data": result}

    def _query_direct(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Answer using the LLM only — no retrieval or document context."""
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        chat_system_prompt = augment_chat_system_prompt(CHAT_SYSTEM_PROMPT, clean_query)

        t_gen_0 = time.time()
        generation_res = self.llm_provider.chat(
            prompt=clean_query,
            chat_history=chat_history,
            system_prompt=chat_system_prompt,
            max_tokens=max_output_tokens,
        )
        gen_latency = round(time.time() - t_gen_0, 4)
        total_latency = round(time.time() - t0, 4)

        metrics_collector.record_query(total_latency, generation_res.get("completion_tokens", 0), success=True)
        QueryTelemetryLogger.log_query_execution(
            query=clean_query,
            retrieved_count=0,
            reranked_count=0,
            latency_breakdown={
                "rewrite": 0.0,
                "retrieval": 0.0,
                "rerank": 0.0,
                "generation": gen_latency,
                "total": total_latency,
            },
            token_usage={
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            model=generation_res.get("model", "unknown"),
            user_roles=roles,
        )

        return {
            "query": clean_query,
            "rewritten_query": None,
            "answer": generation_res["answer"].strip(),
            "citations": [],
            "formatted_citations": "",
            "model": generation_res.get("model"),
            "token_usage": {
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0),
            },
            "latency": {
                "rewrite_seconds": 0.0,
                "retrieval_seconds": 0.0,
                "rerank_seconds": 0.0,
                "generation_seconds": gen_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": 0,
            "reranked_chunks_count": 0,
            "match_percent": 0.0,
            "mode": "direct",
            "cached": False,
            **self._grounding_flags(generation_res["answer"], 0.0, "direct"),
        }
