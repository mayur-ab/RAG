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
from src.retrieval.query_focus import build_retrieval_queries, focus_retrieval_query
from src.retrieval.source_match import find_matching_sources, fetch_chunks_for_sources
from src.retrieval.topic_anchor import (
    TopicAnchor,
    NOT_FOUND_ANSWER,
    apply_relevance_gate,
    build_topic_anchor,
    merge_pinned_sources,
    prioritize_retrieval_results,
    retrieve_pinned_chunks,
    should_reject_off_topic_answer,
)
from src.reranking.identity import IdentityReranker
from src.reranking.cross_encoder import CrossEncoderReranker

from src.context.builder import ContextBuilder
from src.context.citation import CitationFormatter
from src.context.answer_formatter import AnswerFormatter
from src.context.structured_prompt import augment_system_prompt
from src.context.response_length import resolve_max_output_tokens, resolve_context_tokens, parse_requested_words, augment_chat_system_prompt
from src.context.conversation import (
    build_rag_user_message,
    resolve_retrieval_query,
    requires_conversation_only,
    FOLLOWUP_SYSTEM_PROMPT,
)
from src.context.user_memory_messages import (
    is_user_memory_message,
    MEMORY_CONVERSATION_PROMPT,
)
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
from src.memory.manager import MemoryManager, validate_user_id, validate_session_id
from src.memory.profile_store import ProfileStore
from src.memory.session_store import SessionStore
from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.prompt import augment_with_user_memory
from src.context.conversation_context import ConversationContext
from src.context.chat_compact import ChatCompactService, build_compact_user_message
from src.utils.context_budget import build_token_usage
from src.context.document_request import is_long_document_request
from src.generation.long_document import LongDocumentGenerator
from src.classification.metadata_store import DocumentMetadataStore
from src.classification.service import DocumentClassificationService


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
        self.chat_compact_service = ChatCompactService(self.llm_provider)
        self.long_document_generator = LongDocumentGenerator(self)

        self.doc_classification: Optional[DocumentClassificationService] = None
        if settings.ENABLE_DOC_CLASSIFICATION:
            self.doc_classification = DocumentClassificationService(
                metadata_store=DocumentMetadataStore(settings.DOC_CLASSIFICATION_DB_PATH),
                llm_provider=self.llm_provider,
                embedding_provider=self.embedding_provider,
            )

        self.memory_manager: Optional[MemoryManager] = None
        if settings.ENABLE_USER_MEMORY:
            profile_store = ProfileStore(settings.USER_MEMORY_DB_PATH)
            self.memory_manager = MemoryManager(
                profile_store=profile_store,
                session_store=SessionStore(settings.USER_MEMORY_DB_PATH),
                episodic_store=EpisodicMemoryStore(settings.USER_MEMORY_CHROMA_PATH),
                embedding_provider=self.embedding_provider,
                llm_provider=self.llm_provider,
                memory_top_k=settings.USER_MEMORY_TOP_K,
                max_profile_items=settings.USER_MEMORY_MAX_PROFILE_ITEMS,
            )

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

    def list_indexed_documents(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if hasattr(self.vector_store, "list_indexed_sources"):
            indexed = self.vector_store.list_indexed_sources()
        else:
            indexed = []
        if self.doc_classification:
            return self.doc_classification.enrich_indexed_documents(
                indexed,
                category=category,
                subcategory=subcategory,
                tag=tag,
                search=search,
            )
        return indexed

    def _chunk_texts_by_source(self) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for doc in self._all_chunk_documents:
            src = doc.metadata.source_path or doc.metadata.source or ""
            if src:
                grouped.setdefault(src, []).append(doc.content)
        return grouped

    @staticmethod
    def _normalize_user_id(user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        return validate_user_id(user_id)

    @staticmethod
    def _normalize_session_id(session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        return validate_session_id(session_id)

    @staticmethod
    def _token_usage_from_generation(generation_res: Dict[str, Any], *, log: bool = True) -> Dict[str, Any]:
        return build_token_usage(
            generation_res.get("prompt_tokens", 0),
            generation_res.get("completion_tokens", 0),
            model=str(generation_res.get("model", "unknown")),
            log=log,
        )

    @staticmethod
    def _stream_generation_result(llm_provider: Any, answer: str, model: str) -> Dict[str, Any]:
        usage = getattr(llm_provider, "last_stream_usage", None) or {}
        return {
            "answer": answer,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens") or len(answer.split()),
        }

    @staticmethod
    def build_conversation_context(
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_compact: Optional[str] = None,
        routing_turns: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        pinned_sources: Optional[List[str]] = None,
    ) -> ConversationContext:
        return ConversationContext.from_request(
            session_id=session_id,
            chat_id=chat_id,
            chat_compact=chat_compact,
            routing_turns=routing_turns,
            chat_history=chat_history,
            pinned_sources=pinned_sources,
        )

    def _update_chat_compact(self, prior_compact: str, user_message: str, assistant_message: str) -> str:
        if self.memory_manager:
            return self.memory_manager.update_chat_compact(prior_compact, user_message, assistant_message)
        return self.chat_compact_service.update_chat_compact(prior_compact, user_message, assistant_message)

    def _finalize_conversation_result(
        self,
        result: Dict[str, Any],
        conv: ConversationContext,
        clean_query: str,
        answer: str,
        topic_anchor: Optional[TopicAnchor] = None,
    ) -> Dict[str, Any]:
        result["chat_compact"] = self._update_chat_compact(conv.chat_compact, clean_query, answer)
        result["session_id"] = conv.session_id
        result["chat_id"] = conv.chat_id
        citations = result.get("citations") or []
        result["pinned_sources"] = merge_pinned_sources(
            conv.pinned_sources,
            [c.get("source") for c in citations if c.get("source")],
            topic_anchor.topic_terms if topic_anchor else [],
            self.list_indexed_documents(),
        )
        return result

    def _refusal_result(
        self,
        clean_query: str,
        retrieval_query: str,
        rewrite_latency: float,
        ret_latency: float,
        *,
        retrieved_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "query": clean_query,
            "rewritten_query": retrieval_query if retrieval_query != clean_query else None,
            "answer": NOT_FOUND_ANSWER,
            "citations": [],
            "formatted_citations": "",
            "retrieved_context": "",
            "model": self.get_llm_model_name(),
            "token_usage": build_token_usage(0, 0, log=False),
            "latency": {
                "rewrite_seconds": rewrite_latency,
                "retrieval_seconds": ret_latency,
                "rerank_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": round(rewrite_latency + ret_latency, 4),
            },
            "retrieved_chunks_count": retrieved_count,
            "reranked_chunks_count": 0,
            "match_percent": 0.0,
            "mode": "rag",
            "cached": False,
            "grounded": False,
            "not_in_documents": True,
        }

    def _apply_grounding_guard(
        self,
        clean_answer: str,
        match_percent: float,
        topic_anchor: Optional[TopicAnchor],
        context_str: str,
        citations: List[Dict[str, Any]],
    ) -> tuple[str, float, Dict[str, bool]]:
        grounding = self._grounding_flags(clean_answer, match_percent, "rag")
        if topic_anchor and topic_anchor.enforce_topic:
            if should_reject_off_topic_answer(
                clean_answer,
                topic_anchor.topic_terms,
                context_str,
                citations,
            ):
                logger.info(
                    f"Rejected off-topic answer for anchored topics: {topic_anchor.topic_terms}"
                )
                return NOT_FOUND_ANSWER, 0.0, {"grounded": False, "not_in_documents": True}
        return clean_answer, match_percent, grounding

    def _retrieve_with_topic_anchor(
        self,
        clean_query: str,
        conv: ConversationContext,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
        roles: List[str],
    ) -> tuple[str, str, List[tuple], TopicAnchor, bool, float, float]:
        t_rewrite_0 = time.time()
        retrieval_query = resolve_retrieval_query(
            clean_query,
            conv.chat_compact,
            conv.routing_turns,
            self.query_rewriter,
            settings.ENABLE_QUERY_REWRITING,
        )
        rewrite_latency = round(time.time() - t_rewrite_0, 4)

        indexed_sources = self.list_indexed_documents()
        topic_anchor = build_topic_anchor(
            clean_query,
            retrieval_query,
            conv.chat_compact,
            conv.routing_turns,
            conv.pinned_sources,
            indexed_sources,
        )
        effective_query = topic_anchor.anchored_query or retrieval_query

        t_ret_0 = time.time()
        retrieved, rerank_query = self._hybrid_retrieve(
            clean_query=clean_query,
            retrieval_query=effective_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            roles=roles,
            topic_anchor=topic_anchor,
        )
        retrieved = prioritize_retrieval_results(
            retrieved,
            topic_anchor.topic_terms,
            topic_anchor.pinned_sources,
            enforce_pinning=topic_anchor.enforce_pinning,
        )
        retrieved, gate_passed = apply_relevance_gate(retrieved, topic_anchor.topic_terms)

        if not gate_passed and topic_anchor.enforce_topic and self._all_chunk_documents:
            fallback = retrieve_pinned_chunks(
                self._all_chunk_documents,
                topic_anchor.pinned_sources,
                topic_anchor.topic_terms,
            )
            fallback, gate_passed = apply_relevance_gate(fallback, topic_anchor.topic_terms)
            if gate_passed:
                retrieved = fallback
                logger.info(
                    f"Topic anchor re-search via pinned sources: {topic_anchor.pinned_sources}"
                )

        ret_latency = round(time.time() - t_ret_0, 4)
        gate_ok = gate_passed or not topic_anchor.enforce_topic
        return retrieval_query, rerank_query, retrieved, topic_anchor, gate_ok, rewrite_latency, ret_latency

    def _apply_user_memory(self, system_prompt: str, user_id: Optional[str], query: str) -> str:
        if not user_id or not self.memory_manager:
            return system_prompt
        try:
            ctx = self.memory_manager.build_memory_context(user_id, query)
            return augment_with_user_memory(system_prompt, ctx)
        except Exception as exc:
            logger.warning(f"User memory context failed for {user_id}: {exc}")
            return system_prompt

    def _record_user_query(self, user_id: Optional[str], query: str) -> None:
        if not user_id or not self.memory_manager:
            return
        try:
            self.memory_manager.record_query(user_id, query)
        except Exception as exc:
            logger.warning(f"User memory topic record failed for {user_id}: {exc}")

    def _capture_user_memory(
        self,
        user_id: Optional[str],
        query: str,
        routing_turns: Optional[List[Dict[str, str]]],
    ) -> None:
        if not user_id or not self.memory_manager:
            return
        if not is_user_memory_message(query):
            return
        try:
            saved = self.memory_manager.capture_user_statement(user_id, query, routing_turns)
            if saved.get("display_name"):
                logger.info(f"Saved display name for user {user_id}: {saved['display_name']}")
        except Exception as exc:
            logger.warning(f"User memory capture failed for {user_id}: {exc}")

    def start_user_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.memory_manager:
            raise RuntimeError("User memory is disabled")
        uid = self._normalize_user_id(user_id)
        if not uid:
            raise ValueError("Invalid user_id")
        sid = self._normalize_session_id(session_id) if session_id else None
        return self.memory_manager.start_session(uid, sid)

    def end_user_chat(
        self,
        user_id: str,
        session_id: str,
        chat_id: str,
        chat_compact: str,
        *,
        message_count: int = 0,
    ) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"merged": False, "reason": "memory_disabled"}
        uid = self._normalize_user_id(user_id)
        sid = self._normalize_session_id(session_id)
        if not uid or not sid:
            raise ValueError("Invalid user_id or session_id")
        return self.memory_manager.end_chat(
            uid, sid, chat_id, chat_compact, message_count=message_count
        )

    def end_user_session(
        self,
        user_id: str,
        session_id: str,
        final_chat_compact: Optional[str] = None,
        *,
        message_count: int = 0,
        fallback_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"archived": False, "reason": "memory_disabled"}
        uid = self._normalize_user_id(user_id)
        sid = self._normalize_session_id(session_id)
        if not uid or not sid:
            raise ValueError("Invalid user_id or session_id")
        return self.memory_manager.end_session(
            uid,
            sid,
            final_chat_compact,
            message_count=message_count,
            fallback_summary=fallback_summary,
        )

    def archive_user_session(
        self,
        user_id: str,
        chat_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        if not self.memory_manager:
            return {"archived": False, "reason": "memory_disabled", "message_count": 0}
        uid = self._normalize_user_id(user_id)
        if not uid:
            raise ValueError("Invalid user_id")
        return self.memory_manager.archive_session(uid, chat_history or [])

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if not self.memory_manager:
            raise RuntimeError("User memory is disabled")
        uid = self._normalize_user_id(user_id)
        if not uid:
            raise ValueError("Invalid user_id")
        return self.memory_manager.get_profile(uid)

    def delete_user_memory(self, user_id: str) -> Dict[str, Any]:
        if not self.memory_manager:
            raise RuntimeError("User memory is disabled")
        uid = self._normalize_user_id(user_id)
        if not uid:
            raise ValueError("Invalid user_id")
        result = self.memory_manager.delete_all_user_data(uid)
        self._query_cache.clear()
        logger.info(f"Deleted user memory for {uid}: {result}")
        return result

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

    def _hybrid_retrieve(
        self,
        clean_query: str,
        retrieval_query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]],
        roles: List[str],
        topic_anchor: Optional[TopicAnchor] = None,
    ) -> tuple[List[Any], str]:
        """Multi-query hybrid retrieval with filename/title source matching."""
        focused_query = focus_retrieval_query(clean_query)
        search_queries = build_retrieval_queries(clean_query)
        effective_retrieval = retrieval_query
        if topic_anchor and topic_anchor.anchored_query:
            effective_retrieval = topic_anchor.anchored_query
        if effective_retrieval and effective_retrieval not in search_queries:
            search_queries.append(effective_retrieval)

        if topic_anchor and topic_anchor.topic_terms:
            topic_query = " ".join(topic_anchor.topic_terms)
            if topic_query not in search_queries:
                search_queries.append(topic_query)

        merged: Dict[str, tuple] = {}
        per_query_k = max(top_k, 12)

        for query_variant in search_queries:
            for doc, score in self.hybrid_engine.search(
                query=query_variant,
                top_k=per_query_k,
                filter_metadata=filter_metadata,
                allowed_roles=roles,
                use_rrf=True,
            ):
                existing = merged.get(doc.id)
                if existing is None or score > existing[1]:
                    merged[doc.id] = (doc, score)

        if self._all_chunk_documents:
            title_query = focused_query
            if topic_anchor and topic_anchor.topic_terms:
                title_query = " ".join(topic_anchor.topic_terms)
            matched_sources = find_matching_sources(title_query, self.list_indexed_documents())
            if topic_anchor and topic_anchor.pinned_sources:
                for src in topic_anchor.pinned_sources:
                    if src not in matched_sources:
                        matched_sources.append(src)
            if matched_sources:
                logger.info(f"Source title match for '{title_query}': {matched_sources}")
                for doc in fetch_chunks_for_sources(
                    self._all_chunk_documents,
                    matched_sources,
                    max_per_source=15,
                ):
                    existing = merged.get(doc.id)
                    boost = 2.0
                    if topic_anchor and topic_anchor.enforce_pinning:
                        boost = 3.0
                    if existing is None or boost > existing[1]:
                        merged[doc.id] = (doc, boost)

        ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        return ranked[: top_k * 2], focused_query

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

        if self.doc_classification:
            try:
                title = raw_docs[0].metadata.title if raw_docs else os.path.basename(source)
                self.doc_classification.classify_from_chunks(
                    document_id=raw_docs[0].metadata.document_id if raw_docs else resolved_document_id,
                    source=normalized_source,
                    title=title.lstrip("\ufeff").strip() or os.path.basename(source),
                    chunk_texts=[doc.content for doc in chunked_docs[:12]],
                )
            except Exception as exc:
                logger.warning(f"Document classification failed for '{source}': {exc}")

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
        chat_compact: Optional[str] = None,
        routing_turns: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        pinned_sources: Optional[List[str]] = None,
        use_rag: bool = True,
        use_cache: bool = True,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes RAG or direct LLM query pipeline."""
        self.apply_request_model(model)
        model_name = self.get_llm_model_name()
        uid = self._normalize_user_id(user_id)
        conv = self.build_conversation_context(
            session_id=session_id,
            chat_id=chat_id,
            chat_compact=chat_compact,
            routing_turns=routing_turns,
            chat_history=chat_history,
            pinned_sources=pinned_sources,
        )
        if use_cache:
            cached = self._query_cache.get(
                query, use_rag, model_name, conv.routing_turns, uid, conv.chat_id, conv.chat_compact
            )
            if cached:
                return cached

        if not use_rag:
            result = self._query_direct(query=query, conv=conv, user_roles=user_roles, user_id=uid)
        else:
            result = self._query_rag(
                query=query,
                top_k=top_k,
                top_m_rerank=top_m_rerank,
                user_roles=user_roles,
                filter_metadata=filter_metadata,
                conv=conv,
                user_id=uid,
            )

        if use_cache and not result.get("cached"):
            self._query_cache.set(
                query, use_rag, model_name, conv.routing_turns, result, uid, conv.chat_id, conv.chat_compact
            )
        if uid and not result.get("cached"):
            self._record_user_query(uid, result.get("query") or query)
        return result

    def _query_conversation_followup(
        self,
        clean_query: str,
        conv: ConversationContext,
        user_roles: Optional[List[str]],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer follow-ups using compact conversation context — skip retrieval."""
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        self._capture_user_memory(user_id, clean_query, conv.routing_turns)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        base_prompt = MEMORY_CONVERSATION_PROMPT if is_user_memory_message(clean_query) else FOLLOWUP_SYSTEM_PROMPT
        chat_system = augment_chat_system_prompt(base_prompt, clean_query)
        chat_system = self._apply_user_memory(chat_system, user_id, clean_query)
        user_prompt = build_compact_user_message(clean_query, conv.chat_compact)

        t_gen_0 = time.time()
        generation_res = self.llm_provider.chat(
            prompt=user_prompt,
            chat_history=None,
            system_prompt=chat_system,
            max_tokens=max_output_tokens,
        )
        gen_latency = round(time.time() - t_gen_0, 4)
        total_latency = round(time.time() - t0, 4)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])

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
            token_usage=self._token_usage_from_generation(generation_res),
            model=generation_res.get("model", "unknown"),
            user_roles=roles,
        )

        result = {
            "query": clean_query,
            "rewritten_query": None,
            "answer": clean_answer,
            "citations": [],
            "formatted_citations": "",
            "retrieved_context": "",
            "model": generation_res.get("model"),
            "token_usage": self._token_usage_from_generation(generation_res, log=False),
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
            "mode": "rag",
            "cached": False,
            "grounded": False,
            "not_in_documents": False,
        }
        return self._finalize_conversation_result(result, conv, clean_query, clean_answer)

    def _query_rag(
        self,
        query: str,
        top_k: int,
        top_m_rerank: int,
        user_roles: Optional[List[str]],
        filter_metadata: Optional[Dict[str, Any]],
        conv: ConversationContext,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)

        if requires_conversation_only(clean_query, conv.routing_turns, conv.chat_compact):
            return self._query_conversation_followup(clean_query, conv, roles, user_id)

        if is_long_document_request(clean_query):
            logger.info(f"Using hierarchical long-document generation for: {clean_query[:80]}")
            result = self.long_document_generator.generate(
                clean_query=clean_query,
                conv=conv,
                roles=roles,
                filter_metadata=filter_metadata,
                user_id=user_id,
                top_k=top_k,
                top_m_rerank=top_m_rerank,
            )
            return self._finalize_conversation_result(result, conv, clean_query, result["answer"])

        top_k, top_m_rerank = self._retrieval_limits(clean_query, top_k, top_m_rerank)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        context_tokens = resolve_context_tokens(clean_query)

        (
            retrieval_query,
            rerank_query,
            retrieved,
            topic_anchor,
            gate_ok,
            rewrite_latency,
            ret_latency,
        ) = self._retrieve_with_topic_anchor(clean_query, conv, top_k, filter_metadata, roles)

        if not gate_ok:
            logger.info(f"Retrieval relevance gate failed for topics: {topic_anchor.topic_terms}")
            result = self._refusal_result(
                clean_query,
                topic_anchor.anchored_query or retrieval_query,
                rewrite_latency,
                ret_latency,
                retrieved_count=len(retrieved),
            )
            return self._finalize_conversation_result(result, conv, clean_query, result["answer"], topic_anchor)

        t_rank_0 = time.time()
        reranked = self.reranker.rerank(query=rerank_query, documents=retrieved, top_n=top_m_rerank)
        rank_latency = round(time.time() - t_rank_0, 4)

        context_str, citations = self.context_builder.build_context(reranked, max_tokens=context_tokens)
        system_prompt = augment_system_prompt(SYSTEM_PROMPT, clean_query)
        system_prompt = self._apply_user_memory(system_prompt, user_id, clean_query)

        t_gen_0 = time.time()
        generation_res = self.llm_provider.generate(
            prompt=clean_query,
            context=context_str,
            system_prompt=system_prompt,
            max_tokens=max_output_tokens,
            chat_compact=conv.chat_compact,
        )
        gen_latency = round(time.time() - t_gen_0, 4)
        total_latency = round(time.time() - t0, 4)

        citations = AnswerFormatter.normalize_citations(citations)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])
        match_percent = round(EvaluationMetrics.faithfulness(clean_answer, context_str) * 100, 1)
        clean_answer, match_percent, grounding = self._apply_grounding_guard(
            clean_answer,
            match_percent,
            topic_anchor,
            context_str,
            citations,
        )
        formatted_citations = CitationFormatter.format_citations(citations)

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
            token_usage=self._token_usage_from_generation(generation_res),
            model=generation_res.get("model", "unknown"),
            user_roles=roles,
        )

        effective_rewrite = topic_anchor.anchored_query or retrieval_query
        result = {
            "query": clean_query,
            "rewritten_query": effective_rewrite if effective_rewrite != clean_query else None,
            "answer": clean_answer,
            "citations": citations,
            "formatted_citations": formatted_citations,
            "retrieved_context": context_str,
            "model": generation_res.get("model"),
            "token_usage": self._token_usage_from_generation(generation_res, log=False),
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
        return self._finalize_conversation_result(result, conv, clean_query, clean_answer, topic_anchor)

    def query_stream(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        top_m_rerank: int = settings.TOP_K_RERANK,
        user_roles: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_compact: Optional[str] = None,
        routing_turns: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        pinned_sources: Optional[List[str]] = None,
        use_rag: bool = True,
        use_cache: bool = True,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Yield SSE-friendly event dicts for staged streaming responses."""
        self.apply_request_model(model)
        model_name = self.get_llm_model_name()
        uid = self._normalize_user_id(user_id)
        conv = self.build_conversation_context(
            session_id=session_id,
            chat_id=chat_id,
            chat_compact=chat_compact,
            routing_turns=routing_turns,
            chat_history=chat_history,
            pinned_sources=pinned_sources,
        )
        if use_cache:
            cached = self._query_cache.get(
                query, use_rag, model_name, conv.routing_turns, uid, conv.chat_id, conv.chat_compact
            )
            if cached:
                yield {"type": "stage", "stage": "cached", "message": "Returning cached answer..."}
                yield {"type": "token", "content": cached["answer"]}
                yield {"type": "done", "data": cached}
                return

        if not use_rag:
            yield from self._stream_direct(query, conv, user_roles, use_cache, model_name, uid)
            return

        yield from self._stream_rag(
            query, top_k, top_m_rerank, user_roles, filter_metadata, conv, use_cache, model_name, uid
        )

    def _stream_direct(self, query, conv, user_roles, use_cache, model_name, user_id=None):
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)
        self._capture_user_memory(user_id, clean_query, conv.routing_turns)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        base_prompt = MEMORY_CONVERSATION_PROMPT if is_user_memory_message(clean_query) else CHAT_SYSTEM_PROMPT
        chat_system_prompt = augment_chat_system_prompt(base_prompt, clean_query)
        chat_system_prompt = self._apply_user_memory(chat_system_prompt, user_id, clean_query)
        user_prompt = build_compact_user_message(clean_query, conv.chat_compact)
        yield {"type": "stage", "stage": "generating", "message": "Generating answer..."}

        answer_parts: List[str] = []
        stream_fn = getattr(self.llm_provider, "chat_stream", None)
        if stream_fn:
            for token in stream_fn(
                prompt=user_prompt,
                chat_history=None,
                system_prompt=chat_system_prompt,
                max_tokens=max_output_tokens,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts).strip()
            model = f"mock-chat-llm" if settings.LLM_PROVIDER == "mock" else getattr(self.llm_provider, "model", "unknown")
            if settings.LLM_PROVIDER == "ollama":
                model = f"ollama/{self.llm_provider.model}"
            generation_res = self._stream_generation_result(self.llm_provider, answer, model)
        else:
            generation_res = self.llm_provider.chat(
                prompt=user_prompt,
                chat_history=None,
                system_prompt=chat_system_prompt,
                max_tokens=max_output_tokens,
            )
            answer = generation_res["answer"].strip()
            yield {"type": "token", "content": answer}

        total_latency = round(time.time() - t0, 4)
        token_usage = self._token_usage_from_generation(generation_res)
        result = {
            "query": clean_query,
            "rewritten_query": None,
            "answer": answer,
            "citations": [],
            "formatted_citations": "",
            "model": generation_res.get("model"),
            "token_usage": token_usage,
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
        result = self._finalize_conversation_result(result, conv, clean_query, answer)
        if use_cache:
            self._query_cache.set(
                query, False, model_name, conv.routing_turns, result, user_id, conv.chat_id, conv.chat_compact
            )
        if user_id:
            self._record_user_query(user_id, clean_query)
        yield {"type": "done", "data": result}

    def _stream_rag(self, query, top_k, top_m_rerank, user_roles, filter_metadata, conv, use_cache, model_name, user_id=None):
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)

        if requires_conversation_only(clean_query, conv.routing_turns, conv.chat_compact):
            yield {"type": "stage", "stage": "generating", "message": "Continuing conversation..."}
            result = self._query_conversation_followup(clean_query, conv, roles, user_id)
            yield {"type": "token", "content": result["answer"]}
            if use_cache:
                self._query_cache.set(
                    query, True, model_name, conv.routing_turns, result, user_id, conv.chat_id, conv.chat_compact
                )
            if user_id:
                self._record_user_query(user_id, clean_query)
            yield {"type": "done", "data": result}
            return

        if is_long_document_request(clean_query):
            logger.info(f"Streaming hierarchical long-document generation for: {clean_query[:80]}")
            result = None
            for event in self.long_document_generator.generate_stream(
                clean_query=clean_query,
                conv=conv,
                roles=roles,
                filter_metadata=filter_metadata,
                user_id=user_id,
                top_k=top_k,
                top_m_rerank=top_m_rerank,
            ):
                if event.get("type") == "done":
                    result = self._finalize_conversation_result(
                        event["data"], conv, clean_query, event["data"]["answer"]
                    )
                    yield {"type": "done", "data": result}
                else:
                    yield event
            if result and use_cache:
                self._query_cache.set(
                    query, True, model_name, conv.routing_turns, result, user_id, conv.chat_id, conv.chat_compact
                )
            if user_id and result:
                self._record_user_query(user_id, clean_query)
            return

        yield {"type": "stage", "stage": "rewriting", "message": "Rewriting question..."}
        top_k, top_m_rerank = self._retrieval_limits(clean_query, top_k, top_m_rerank)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        context_tokens = resolve_context_tokens(clean_query)

        (
            retrieval_query,
            rerank_query,
            retrieved,
            topic_anchor,
            gate_ok,
            rewrite_latency,
            ret_latency,
        ) = self._retrieve_with_topic_anchor(clean_query, conv, top_k, filter_metadata, roles)

        if not gate_ok:
            logger.info(f"Retrieval relevance gate failed for topics: {topic_anchor.topic_terms}")
            result = self._refusal_result(
                clean_query,
                topic_anchor.anchored_query or retrieval_query,
                rewrite_latency,
                ret_latency,
                retrieved_count=len(retrieved),
            )
            result = self._finalize_conversation_result(result, conv, clean_query, result["answer"], topic_anchor)
            yield {"type": "token", "content": result["answer"]}
            if use_cache:
                self._query_cache.set(
                    query, True, model_name, conv.routing_turns, result, user_id, conv.chat_id, conv.chat_compact
                )
            if user_id:
                self._record_user_query(user_id, clean_query)
            yield {"type": "done", "data": result}
            return

        yield {"type": "stage", "stage": "searching", "message": "Searching documents..."}
        yield {"type": "stage", "stage": "reading", "message": f"Reading {len(retrieved)} documents..."}
        reranked = self.reranker.rerank(query=rerank_query, documents=retrieved, top_n=top_m_rerank)
        context_str, citations = self.context_builder.build_context(reranked, max_tokens=context_tokens)
        system_prompt = augment_system_prompt(SYSTEM_PROMPT, clean_query)
        system_prompt = self._apply_user_memory(system_prompt, user_id, clean_query)

        yield {"type": "stage", "stage": "generating", "message": "Generating answer..."}
        answer_parts: List[str] = []
        stream_fn = getattr(self.llm_provider, "generate_stream", None)
        if stream_fn:
            for token in stream_fn(
                prompt=clean_query,
                context=context_str,
                system_prompt=system_prompt,
                max_tokens=max_output_tokens,
                chat_compact=conv.chat_compact,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts).strip()
            model = f"ollama/{self.llm_provider.model}" if settings.LLM_PROVIDER == "ollama" else "unknown"
            generation_res = self._stream_generation_result(self.llm_provider, answer, model)
        else:
            generation_res = self.llm_provider.generate(
                prompt=clean_query,
                context=context_str,
                system_prompt=system_prompt,
                max_tokens=max_output_tokens,
                chat_compact=conv.chat_compact,
            )
            answer = generation_res["answer"]
            yield {"type": "token", "content": answer}

        total_latency = round(time.time() - t0, 4)
        citations = AnswerFormatter.normalize_citations(citations)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])
        match_percent = round(EvaluationMetrics.faithfulness(clean_answer, context_str) * 100, 1)
        clean_answer, match_percent, grounding = self._apply_grounding_guard(
            clean_answer,
            match_percent,
            topic_anchor,
            context_str,
            citations,
        )
        token_usage = self._token_usage_from_generation(generation_res)
        result = {
            "query": clean_query,
            "rewritten_query": (topic_anchor.anchored_query or retrieval_query)
            if (topic_anchor.anchored_query or retrieval_query) != clean_query
            else None,
            "answer": clean_answer,
            "citations": citations,
            "formatted_citations": CitationFormatter.format_citations(citations),
            "retrieved_context": context_str,
            "model": generation_res.get("model"),
            "token_usage": token_usage,
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
            **grounding,
        }
        result = self._finalize_conversation_result(result, conv, clean_query, clean_answer, topic_anchor)
        if use_cache:
            self._query_cache.set(
                query, True, model_name, conv.routing_turns, result, user_id, conv.chat_id, conv.chat_compact
            )
        if user_id:
            self._record_user_query(user_id, clean_query)
        yield {"type": "done", "data": result}

    def _query_direct(
        self,
        query: str,
        conv: ConversationContext,
        user_roles: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer using the LLM only — no retrieval or document context."""
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES
        clean_query = InputValidator.validate_query(query)
        self._capture_user_memory(user_id, clean_query, conv.routing_turns)
        max_output_tokens = resolve_max_output_tokens(clean_query)
        base_prompt = MEMORY_CONVERSATION_PROMPT if is_user_memory_message(clean_query) else CHAT_SYSTEM_PROMPT
        chat_system_prompt = augment_chat_system_prompt(base_prompt, clean_query)
        chat_system_prompt = self._apply_user_memory(chat_system_prompt, user_id, clean_query)
        user_prompt = build_compact_user_message(clean_query, conv.chat_compact)

        t_gen_0 = time.time()
        generation_res = self.llm_provider.chat(
            prompt=user_prompt,
            chat_history=None,
            system_prompt=chat_system_prompt,
            max_tokens=max_output_tokens,
        )
        gen_latency = round(time.time() - t_gen_0, 4)
        total_latency = round(time.time() - t0, 4)
        clean_answer = generation_res["answer"].strip()

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
            token_usage=self._token_usage_from_generation(generation_res),
            model=generation_res.get("model", "unknown"),
            user_roles=roles,
        )

        result = {
            "query": clean_query,
            "rewritten_query": None,
            "answer": clean_answer,
            "citations": [],
            "formatted_citations": "",
            "model": generation_res.get("model"),
            "token_usage": self._token_usage_from_generation(generation_res, log=False),
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
            **self._grounding_flags(clean_answer, 0.0, "direct"),
        }
        return self._finalize_conversation_result(result, conv, clean_query, clean_answer)
