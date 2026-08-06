import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.service import RAGPipelineService


@pytest.fixture
def ingest_service(tmp_path, monkeypatch):
    db_path = tmp_path / "dedup_chroma"
    monkeypatch.setenv("VECTOR_DB_PATH", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("RERANKER_PROVIDER", "identity")

    from config.settings import Settings

    monkeypatch.setattr("config.settings.settings", Settings())
    from importlib import reload
    import config.settings as settings_mod
    reload(settings_mod)

    return RAGPipelineService()


def test_ingest_skips_unchanged_file(ingest_service, tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("Hello from dedup test document.", encoding="utf-8")
    source = str(doc).replace("\\", "/")

    first = ingest_service.ingest_source(source=source, chunking_strategy="recursive")
    assert first["skipped"] is False
    assert first["total_chunks"] >= 1

    stats_after_first = ingest_service.vector_store.get_stats()["total_vectors"]

    second = ingest_service.ingest_source(source=source, chunking_strategy="recursive")
    assert second["skipped"] is True
    assert second["status"] == "skipped"
    assert second["total_chunks"] == first["total_chunks"]

    stats_after_second = ingest_service.vector_store.get_stats()["total_vectors"]
    assert stats_after_second == stats_after_first


def test_ingest_replaces_modified_file(ingest_service, tmp_path):
    doc = tmp_path / "mutable.txt"
    doc.write_text("Version one content here.", encoding="utf-8")
    source = str(doc).replace("\\", "/")

    first = ingest_service.ingest_source(source=source, chunking_strategy="recursive")
    vectors_after_first = ingest_service.vector_store.get_stats()["total_vectors"]

    doc.write_text("Version two with completely different content.", encoding="utf-8")
    second = ingest_service.ingest_source(source=source, chunking_strategy="recursive")
    assert second["skipped"] is False
    assert second["status"] == "replaced"

    vectors_after_second = ingest_service.vector_store.get_stats()["total_vectors"]
    assert vectors_after_second == vectors_after_first


def test_ingest_force_reindexes(ingest_service, tmp_path):
    doc = tmp_path / "force.txt"
    doc.write_text("Force reindex test.", encoding="utf-8")
    source = str(doc).replace("\\", "/")

    ingest_service.ingest_source(source=source, chunking_strategy="recursive")
    vectors_before = ingest_service.vector_store.get_stats()["total_vectors"]

    forced = ingest_service.ingest_source(
        source=source,
        chunking_strategy="recursive",
        skip_if_unchanged=False,
        force=True,
    )
    assert forced["skipped"] is False
    vectors_after = ingest_service.vector_store.get_stats()["total_vectors"]
    assert vectors_after == vectors_before
