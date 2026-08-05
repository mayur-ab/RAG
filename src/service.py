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
from src.reranking.identity import IdentityReranker
from src.reranking.cross_encoder import CrossEncoderReranker

from src.context.builder import ContextBuilder
from src.context.citation import CitationFormatter
from src.context.answer_formatter import AnswerFormatter

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
        self.context_builder = ContextBuilder(max_tokens=4000)

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

        self._all_chunk_documents: List[Document] = []
        self._rehydrate_bm25_index()

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

    def ingest_source(
        self,
        source: str,
        document_id: Optional[str] = None,
        chunking_strategy: str = settings.CHUNKING_STRATEGY,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        allowed_roles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Ingests a file or URL into the RAG system."""
        t0 = time.time()
        loader = self._get_loader(source)
        raw_docs = loader.process(source=source, document_id=document_id)

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

        logger.info(f"Ingested '{source}': {len(chunked_docs)} chunks created in {elapsed}s.")
        return {
            "source": source,
            "document_id": raw_docs[0].metadata.document_id if raw_docs else document_id,
            "total_chunks": len(chunked_docs),
            "chunking_strategy": chunking_strategy,
            "elapsed_seconds": elapsed
        }

    def query(
        self,
        query: str,
        top_k: int = settings.TOP_K_RETRIEVAL,
        top_m_rerank: int = settings.TOP_K_RERANK,
        user_roles: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes full RAG query pipeline: Validate -> Hybrid Retrieval -> Rerank -> Context -> LLM -> Citation."""
        t0 = time.time()
        roles = user_roles or settings.DEFAULT_USER_ROLES

        # 1. Input Sanitization
        clean_query = InputValidator.validate_query(query)

        # 2. Hybrid Search
        t_ret_0 = time.time()
        retrieved = self.hybrid_engine.search(
            query=clean_query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            allowed_roles=roles,
            use_rrf=True
        )
        ret_latency = round(time.time() - t_ret_0, 4)

        # 3. Re-ranking
        t_rank_0 = time.time()
        reranked = self.reranker.rerank(query=clean_query, documents=retrieved, top_n=top_m_rerank)
        rank_latency = round(time.time() - t_rank_0, 4)

        # 4. Context Construction
        context_str, citations = self.context_builder.build_context(reranked)

        # 5. LLM Generation
        t_gen_0 = time.time()
        generation_res = self.llm_provider.generate(prompt=clean_query, context=context_str)
        gen_latency = round(time.time() - t_gen_0, 4)

        total_latency = round(time.time() - t0, 4)

        citations = AnswerFormatter.normalize_citations(citations)
        clean_answer = AnswerFormatter.sanitize_answer(generation_res["answer"])
        formatted_citations = CitationFormatter.format_citations(citations)
        match_percent = round(
            EvaluationMetrics.faithfulness(clean_answer, context_str) * 100, 1
        )

        # Telemetry & Metrics logging
        metrics_collector.record_query(total_latency, generation_res.get("completion_tokens", 0), success=True)
        QueryTelemetryLogger.log_query_execution(
            query=clean_query,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            latency_breakdown={"retrieval": ret_latency, "rerank": rank_latency, "generation": gen_latency, "total": total_latency},
            token_usage={"prompt_tokens": generation_res.get("prompt_tokens", 0), "completion_tokens": generation_res.get("completion_tokens", 0)},
            model=generation_res.get("model", "unknown"),
            user_roles=roles
        )

        return {
            "query": clean_query,
            "answer": clean_answer,
            "citations": citations,
            "formatted_citations": formatted_citations,
            "model": generation_res.get("model"),
            "token_usage": {
                "prompt_tokens": generation_res.get("prompt_tokens", 0),
                "completion_tokens": generation_res.get("completion_tokens", 0)
            },
            "latency": {
                "retrieval_seconds": ret_latency,
                "rerank_seconds": rank_latency,
                "generation_seconds": gen_latency,
                "total_seconds": total_latency
            },
            "retrieved_chunks_count": len(retrieved),
            "reranked_chunks_count": len(reranked),
            "match_percent": match_percent,
        }
