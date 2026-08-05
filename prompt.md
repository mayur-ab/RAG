You are a Senior AI Engineer and System Architect.

Your task is to design and implement a production-ready Retrieval-Augmented Generation (RAG) system.

Before writing code, analyze requirements, architecture, scalability, retrieval quality, observability, and security considerations.

Follow these engineering principles:

# Objective

Build a RAG system that provides accurate, grounded, low-hallucination answers from a knowledge base.

The system must be modular, scalable, maintainable, and production-ready.

# Core Requirements

Implement the complete RAG pipeline:

1. Document Ingestion
2. Document Parsing
3. Text Cleaning
4. Chunking
5. Metadata Extraction
6. Embedding Generation
7. Vector Storage
8. Retrieval
9. Hybrid Search
10. Re-ranking
11. Context Construction
12. LLM Generation
13. Source Citation
14. Evaluation
15. Monitoring

# Architecture Requirements

Design the system using clean architecture principles.

Separate layers:

* API Layer
* Service Layer
* Retrieval Layer
* Embedding Layer
* Vector Database Layer
* LLM Layer
* Evaluation Layer
* Monitoring Layer

Code should be modular and independently testable.

# Data Ingestion Requirements

Support:

* PDF
* DOCX
* TXT
* Markdown
* HTML
* Web pages

Create a pluggable document loader architecture.

Each loader should expose:

load()
parse()
extract_metadata()

# Chunking Requirements

Implement configurable chunking strategies:

* Fixed-size chunking
* Recursive chunking
* Semantic chunking
* Header-aware chunking

Chunk configuration:

* chunk_size
* chunk_overlap
* separator strategy

Ensure chunks preserve semantic meaning.

Avoid splitting concepts across chunks whenever possible.

# Metadata Requirements

Every chunk must store:

* document_id
* source
* title
* section
* author
* created_at
* updated_at
* tags
* version

Metadata must be searchable and filterable.

# Embedding Requirements

Create an abstraction layer for embeddings.

Support:

* OpenAI embeddings
* BGE embeddings
* E5 embeddings
* Future embedding providers

Embedding providers must be swappable without changing business logic.

# Vector Database Requirements

Create a vector store abstraction.

Support:

* FAISS
* Chroma
* Qdrant
* Pinecone

Design the interface so databases can be swapped easily.

# Retrieval Requirements

Implement:

1. Vector similarity search
2. Keyword search (BM25)
3. Hybrid retrieval

Allow configurable Top-K retrieval.

Support metadata filtering.

# Re-ranking Requirements

Implement a reranking layer.

Support:

* Cross Encoder rerankers
* BGE rerankers
* Future reranking models

Workflow:

Retrieve N chunks
→ Rerank
→ Return best M chunks

# Context Construction

Implement context builder.

Requirements:

* Deduplicate chunks
* Preserve ordering
* Respect token limits
* Include citations
* Prevent context overflow

Output should be optimized for LLM consumption.

# Prompting Requirements

System prompt:

"Answer only using the provided context.

If the answer cannot be found in the context, explicitly say:
'I could not find that information in the provided knowledge base.'

Do not fabricate information.

Always cite sources."

# LLM Layer

Support multiple providers:

* OpenAI
* Anthropic
* Gemini
* Local models

Use an abstraction layer.

Allow model switching through configuration.

# Citation Requirements

Every answer must include:

* Source document
* Section
* Chunk reference

Citations must be traceable.

# Evaluation Framework

Implement automated evaluation.

Metrics:

* Retrieval Precision
* Retrieval Recall
* Context Relevance
* Answer Relevance
* Faithfulness
* Hallucination Rate

Provide benchmark scripts.

# Monitoring Requirements

Log:

* User query
* Retrieved chunks
* Similarity scores
* Rerank scores
* Generated answer
* Latency metrics
* Token usage
* Errors

Provide structured logging.

# Security Requirements

Implement:

* Role-based access control
* Metadata-based permissions
* User-level document access
* Secret management
* Input validation

Never expose restricted documents.

# Scalability Requirements

Design for:

* Millions of chunks
* Concurrent users
* Horizontal scaling
* Incremental indexing
* Batch ingestion

# API Requirements

Create REST APIs:

POST /ingest

POST /query

POST /evaluate

GET /documents

GET /health

GET /metrics

Include request and response schemas.

# Testing Requirements

Generate:

* Unit tests
* Integration tests
* Retrieval tests
* Evaluation tests

Coverage target: 80%+

# Code Quality Requirements

Before implementing:

1. Explain architecture.
2. Explain design decisions.
3. Identify trade-offs.
4. Identify bottlenecks.
5. Propose folder structure.

Then generate production-ready code.

Avoid toy examples.

Avoid shortcuts.

Avoid placeholder implementations unless explicitly marked.

Use type hints, documentation, configuration management, dependency injection, logging, and clean coding practices.

For every major component:

* Explain why it exists.
* Explain how it interacts with the system.
* Explain scaling considerations.

Act as a senior engineer building a system that will be deployed in production and maintained by a team for years.
