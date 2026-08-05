import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.schema import Document, ChunkMetadata
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveCharacterChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.header_aware import HeaderAwareChunker


def test_chunkers():
    meta = ChunkMetadata(document_id="doc1", source="test.txt", title="Test Doc")
    long_text = "Paragraph one content. " * 30 + "\n\n" + "Paragraph two content. " * 30
    doc = Document(content=long_text, metadata=meta)

    # Fixed Size
    fixed = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
    fixed_chunks = fixed.chunk([doc])
    assert len(fixed_chunks) > 1

    # Recursive
    rec = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)
    rec_chunks = rec.chunk([doc])
    assert len(rec_chunks) > 1

    # Semantic
    sem = SemanticChunker(chunk_size=300, chunk_overlap=50)
    sem_chunks = sem.chunk([doc])
    assert len(sem_chunks) > 1

    # Header Aware
    hdr_doc = Document(content="# Part 1: Intro\nHello world.\n# Part 2: Body\nDeep details.", metadata=meta)
    hdr = HeaderAwareChunker(chunk_size=500)
    hdr_chunks = hdr.chunk([hdr_doc])
    assert len(hdr_chunks) == 2
    assert hdr_chunks[0].metadata.section == "Part 1: Intro"
