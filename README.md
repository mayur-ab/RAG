# Scalable Enterprise RAG Architecture

A production-ready, highly scalable, modular Retrieval-Augmented Generation (RAG) system built with Clean Architecture principles. Designed for zero-cost local deployment using Ollama, Qdrant/Chroma, or pure in-memory Python, with optional cloud acceleration via NVIDIA NIM free API, OpenAI, or Gemini.

---

## Key Features

1. **Clean Layered Architecture**: Strict separation between API, Ingestion, Chunking, Metadata, Embeddings, Vector Stores, Hybrid Retrieval, Reranking, Context Building, LLM Providers, Security, Evaluation, and Observability.
2. **Pluggable Document Ingestion**: Built-in support for PDF, DOCX, Plain Text, Markdown, HTML, and Web URLs with automatic metadata extraction.
3. **Configurable Chunking Strategies**:
   - Fixed-size character chunking with overlap
   - Hierarchical recursive character chunking
   - Sentence-bounded semantic chunking
   - Markdown & document header-aware chunking
4. **Swappable Embedding Abstraction**: Supports Ollama (`nomic-embed-text`), SentenceTransformers (`bge-small-en-v1.5`), OpenAI (`text-embedding-3-small`), and deterministic mock embeddings.
5. **Swappable Vector Databases**: Qdrant, ChromaDB, FAISS, and high-performance native Python Memory Vector Store with cosine similarity and disk persistence.
6. **Hybrid Retrieval + Reciprocal Rank Fusion (RRF)**: Combines Dense Vector Similarity Search with Sparse Keyword Search (BM25) using RRF and Min-Max Score Fusion.
7. **Re-ranking Layer**: CrossEncoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and score-based identity fallbacks.
8. **Multi-LLM Provider Integration**:
   - **Local**: Ollama (`llama3.1:8b`, `qwen2.5`)
   - **Free Cloud API**: NVIDIA NIM (`integrate.api.nvidia.com` with `deepseek-ai/deepseek-v4-flash` or `meta/llama-3.3-70b-instruct`)
   - **Paid Cloud APIs**: OpenAI (`gpt-4o-mini`), Gemini (`gemini-1.5-flash`)
   - **Offline Mock**: Grounded mock LLM for fast unit testing.
9. **Strict Prompt Engineering & Traceable Citations**: Enforces zero hallucination and traceable source document, section, and chunk citations.
10. **RBAC & Security Filter**: Metadata-based document security filtering and prompt injection detection.
11. **Automated Evaluation Framework**: Computes Retrieval Precision, Retrieval Recall, Context Relevance, Faithfulness, and Hallucination Rate.
12. **Structured Logging & Telemetry**: JSON log formatter with latency breakdown tracking and `/metrics` endpoint.

---

## Directory Structure

```
.
├── config/
│   ├── settings.py           # Pydantic configuration settings
│   └── logging_config.py     # JSON logging & telemetry setup
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI entrypoint
│   │   ├── schemas.py        # Pydantic API models
│   │   ├── dependencies.py   # FastAPI dependency injectors
│   │   └── routes/           # /ingest, /query, /evaluate, /health, /metrics
│   ├── ingestion/            # Document loaders (PDF, DOCX, TXT, HTML, Web)
│   ├── chunking/             # Fixed, Recursive, Semantic, Header-aware strategies
│   ├── metadata/             # Schema & automatic metadata extraction
│   ├── embeddings/           # Ollama, OpenAI, SentenceTransformers, Mock providers
│   ├── vector_store/         # MemoryStore, Qdrant, Chroma, FAISS implementations
│   ├── retrieval/            # Vector, BM25, Hybrid RRF engines & Security filter
│   ├── reranking/            # CrossEncoder & Identity rerankers
│   ├── context/              # Context builder & citation formatter
│   ├── llm/                  # Ollama, NVIDIA NIM, OpenAI, Gemini, Mock providers
│   ├── evaluation/           # RAG quality metrics & evaluator
│   ├── monitoring/           # Telemetry metrics collector & logger
│   ├── security/             # RBAC engine & input validator
│   └── service.py            # Central RAG pipeline service orchestrator
├── scripts/
│   ├── ingest_temp_doc.py    # Temp-Doc ingestion script
│   └── run_benchmark.py      # Automated benchmark script
├── tests/                    # Comprehensive pytest suite
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Ingest Temp-Doc Knowledge Base

Run the pre-configured script to index `Temp-Doc/The Rishi Cognitive Framework_ Ancient Logic for Modern Minds.txt`:

```bash
python scripts/ingest_temp_doc.py
```

### 2. Run Benchmark Evaluation

Run the automated evaluation benchmark:

```bash
python scripts/run_benchmark.py
```

### 3. Run Test Suite

Run unit and integration tests using `pytest`:

```bash
pytest tests/
```

### 4. Start REST API Server

Start the FastAPI application:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI documentation is available at `http://localhost:8000/docs`.

---

## REST API Endpoints

- **`POST /ingest`**: Ingest a document file or web URL into the vector database.
- **`POST /query`**: Issue a grounded RAG query with citations, reranking, and token usage telemetry.
- **`POST /evaluate`**: Run automated evaluation metrics against query results.
- **`GET /documents`**: Retrieve index statistics and document count.
- **`GET /health`**: Health check & provider state.
- **`GET /metrics`**: Query telemetry counters and average latency.
