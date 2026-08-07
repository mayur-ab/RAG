With **350 documents**, you're moving from a simple RAG to a **Knowledge Management System**.

The mistake many teams make is:

```text
Upload PDFs
↓
Chunk
↓
Embed
↓
Store in ChromaDB
↓
Show giant file list
```

At 20 docs this works.

At 350 docs users start asking:

* "Show HR policies"
* "Show API docs"
* "Show invoices"
* "Show architecture documents"
* "Show onboarding documents"

and scrolling becomes unusable.

---

# Recommended Architecture

```text
Document
    ↓
Ingestion
    ↓
Classification
    ↓
Metadata Storage
    ↓
ChromaDB
    ↓
UI Filters
```

Store metadata separately from embeddings.

Example:

```json
{
  "doc_id": "123",
  "file_name": "rag_architecture_v2.pdf",
  "category": "Technical",
  "subcategory": "RAG",
  "department": "AI",
  "tags": [
    "chromadb",
    "retrieval",
    "embeddings"
  ]
}
```

---

# Approach 1: LLM-based Auto Classification (Best)

During ingestion:

```text
Document
↓
Extract text
↓
LLM
↓
Generate category
↓
Save metadata
↓
Embed
```

Prompt:

```text
Classify this document.

Categories:
- HR
- Finance
- Legal
- Technical
- Operations
- Sales
- Marketing

Return JSON only.

Document:
{first 3000 chars}
```

Output:

```json
{
  "category": "Technical",
  "subcategory": "RAG",
  "tags": [
    "vector database",
    "chromadb",
    "retrieval"
  ]
}
```

Store alongside Chroma.

---

# Approach 2: Embedding Clustering (Very Useful)

If you don't know categories beforehand.

Take document embeddings:

```text
Doc1 Embedding
Doc2 Embedding
Doc3 Embedding
...
Doc350 Embedding
```

Run:

```text
K-Means
or
HDBSCAN
```

Result:

```text
Cluster 1
  RAG docs

Cluster 2
  HR docs

Cluster 3
  Finance docs

Cluster 4
  Architecture docs
```

Then ask LLM:

```text
These files belong together:

file1.pdf
file2.pdf
file3.pdf

Give a cluster name.
```

Output:

```text
RAG & Retrieval
```

Now you have categories automatically.

---

# Approach 3: Hybrid (Production Recommendation)

Use both.

### During Ingestion

LLM generates:

```text
Category
Subcategory
Tags
```

### Periodically

Run clustering:

```text
350 docs
↓
Group Similar Docs
↓
Create Collections
```

This discovers hidden relationships.

---

# Suggested Metadata Structure

```json
{
  "doc_id": "123",

  "title": "RAG Architecture",

  "category": "Technical",

  "subcategory": "AI",

  "tags": [
    "rag",
    "chromadb",
    "llm"
  ],

  "summary": "Architecture for local RAG system.",

  "created_at": "2026-08-07"
}
```

---

# UI Design

Instead of:

```text
350 files
```

Show:

```text
📁 Technical (120)

📁 HR (40)

📁 Finance (35)

📁 Legal (25)

📁 Operations (50)

📁 Uncategorized (80)
```

Click Technical:

```text
📁 AI
📁 Backend
📁 Infrastructure
📁 Architecture
```

Click AI:

```text
RAG Architecture.pdf
Vector DB Guide.pdf
Embedding Strategy.pdf
...
```

---

# Smart Tag System

Generate tags automatically:

```text
Document
↓
LLM
↓
Tags
```

Example:

```json
[
  "rag",
  "chromadb",
  "retrieval",
  "embeddings"
]
```

UI:

```text
#rag
#chromadb
#retrieval
#embeddings
```

Users can filter instantly.

---

# Semantic Collections (Very Powerful)

Instead of fixed folders:

```text
Technical
Finance
HR
```

Create dynamic collections.

User asks:

> Show all documents related to AI.

Search document-level embeddings:

```text
Query:
AI
```

Return:

```text
RAG Architecture.pdf
LLM Evaluation.pdf
Prompt Engineering.pdf
Embedding Guide.pdf
```

Even if they're in different folders.

---

# What I would build for 350 docs

```text
Ingestion Pipeline
    ↓
Extract Text
    ↓
Generate Summary
    ↓
Generate Category
    ↓
Generate Subcategory
    ↓
Generate Tags
    ↓
Store Metadata (SQLite/Postgres)
    ↓
Store Embeddings (ChromaDB)
```

UI:

```text
Browse By:
✓ Category
✓ Tags
✓ Search
✓ Semantic Collections
✓ Recent Documents
✓ Most Accessed Documents
```

This scales comfortably from **350 documents to several thousand** without changing the architecture. The key principle is:

> **Use ChromaDB for semantic retrieval, and use metadata (category, tags, summaries) for organization and navigation.** Don't rely on embeddings alone to organize a document library.
