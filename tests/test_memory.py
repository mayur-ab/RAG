import pytest

from src.llm.mock import MockLLMProvider
from src.embeddings.mock import MockEmbeddingProvider
from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.extractor import MemoryExtractor, extract_topics_from_query, format_conversation
from src.memory.manager import MemoryManager, validate_user_id
from src.memory.models import UserMemoryContext, UserPreferences, UserProfile, UserStyle
from src.memory.profile_store import ProfileStore
from src.memory.session_store import SessionStore
from src.memory.prompt import augment_with_user_memory, format_memory_context


@pytest.fixture
def temp_profile_store(tmp_path):
    return ProfileStore(str(tmp_path / "profiles.db"))


@pytest.fixture
def temp_session_store(tmp_path):
    return SessionStore(str(tmp_path / "sessions.db"))


@pytest.fixture
def temp_episodic_store(tmp_path):
    return EpisodicMemoryStore(str(tmp_path / "episodic_chroma"))


@pytest.fixture
def memory_manager(temp_profile_store, temp_session_store, temp_episodic_store):
    return MemoryManager(
        profile_store=temp_profile_store,
        session_store=temp_session_store,
        episodic_store=temp_episodic_store,
        embedding_provider=MockEmbeddingProvider(),
        llm_provider=MockLLMProvider(),
        memory_top_k=3,
    )


def test_validate_user_id():
    assert validate_user_id("user-abc-12345") == "user-abc-12345"
    with pytest.raises(ValueError):
        validate_user_id("short")
    with pytest.raises(ValueError):
        validate_user_id("bad id!")


def test_profile_store_create_and_merge(temp_profile_store):
    profile = temp_profile_store.get_profile("test-user-001")
    assert profile.user_id == "test-user-001"

    updated = temp_profile_store.merge_profile_updates(
        "test-user-001",
        preferences=["likes detailed explanations"],
        interests=["RAG", "ChromaDB"],
        projects=["Local RAG System"],
        facts=["uses Ollama"],
        style_updates={"likes_examples": True, "experience_level": "AI Engineer"},
    )
    assert updated.preferences.response_style == "detailed"
    assert "RAG" in updated.interests
    assert updated.style.experience_level == "AI Engineer"

    temp_profile_store.record_topics("test-user-001", ["chromadb", "memory"])
    assert "chromadb" in temp_profile_store.top_topics("test-user-001")


def test_memory_manager_capture_user_statement(memory_manager):
    user_id = "capture-user-001"
    saved = memory_manager.capture_user_statement(user_id, "hey my name is mayur", None)
    assert saved["display_name"] == "Mayur"
    profile = memory_manager.get_profile(user_id)
    assert profile["display_name"] == "Mayur"
    assert any("Mayur" in fact for fact in profile["facts"])


def test_format_memory_context_includes_profile():
    ctx = UserMemoryContext(
        profile=UserProfile(
            user_id="test-user-001",
            preferences=UserPreferences(response_style="detailed"),
            projects=["Local RAG System"],
            interests=["ChromaDB"],
            style=UserStyle(likes_examples=True, experience_level="AI Engineer"),
        ),
        frequent_topics=["rag", "memory"],
    )
    block = format_memory_context(ctx)
    assert "User Profile" in block
    assert "Local RAG System" in block
    assert "ChromaDB" in block
    assert "Frequently asked topics" in block


def test_augment_with_user_memory_appends_to_prompt():
    ctx = UserMemoryContext(
        profile=UserProfile(user_id="test-user-001", projects=["Local RAG"]),
    )
    augmented = augment_with_user_memory("Base prompt.", ctx)
    assert augmented.startswith("Base prompt.")
    assert "Local RAG" in augmented


def test_memory_manager_archive_session(memory_manager):
    user_id = "archive-user-001"
    history = [
        {"role": "user", "content": "How do I build a local RAG with ChromaDB?"},
        {"role": "assistant", "content": "Start with ingestion and embeddings."},
        {"role": "user", "content": "I want long-term memory too."},
    ]
    result = memory_manager.archive_session(user_id, history)
    assert result["archived"] is True
    assert result["summary"]
    profile = memory_manager.get_profile(user_id)
    assert profile["active_session"]["working_compact"]


def test_memory_manager_build_context_after_archive(memory_manager):
    user_id = "context-user-001"
    session = memory_manager.start_session(user_id)
    memory_manager.end_chat(
        user_id,
        session["session_id"],
        "chat-001",
        "User asked about RAG memory architecture; assistant explained SQLite profiles and Chroma summaries.",
    )
    memory_manager.end_session(user_id, session["session_id"])
    ctx = memory_manager.build_memory_context(user_id, "How should I improve my RAG?")
    block = format_memory_context(ctx)
    assert block


def test_extract_topics_from_query():
    topics = extract_topics_from_query("How should I improve my RAG pipeline with ChromaDB?")
    assert "rag" in topics or "chromadb" in topics


def test_memory_manager_session_hierarchy(memory_manager):
    user_id = "session-user-001"
    session = memory_manager.start_session(user_id)
    session_id = session["session_id"]
    chat_id = "chat-thread-001"

    end_chat = memory_manager.end_chat(
        user_id,
        session_id,
        chat_id,
        "User asked about RAG memory; assistant explained SQLite profiles and Chroma episodic storage.",
    )
    assert end_chat["merged"] is True

    archived = memory_manager.end_session(user_id, session_id)
    assert archived["archived"] is True

    profile = memory_manager.get_profile(user_id)
    assert profile["active_session"] is None
    assert profile["recent_sessions"]
    assert profile["recent_sessions"][0]["summary"]


def test_end_chat_counts_activity_without_compact(memory_manager):
    user_id = "activity-user-002"
    session = memory_manager.start_session(user_id)
    end_chat = memory_manager.end_chat(
        user_id,
        session["session_id"],
        "chat-no-compact",
        "",
        message_count=4,
    )
    assert end_chat["merged"] is True
    assert end_chat["chat_count"] == 1

    archived = memory_manager.end_session(
        user_id,
        session["session_id"],
        message_count=4,
        fallback_summary="Discussed Duolingo strategy and product roadmap.",
    )
    assert archived["archived"] is True
    assert "Duolingo" in archived["summary"]


def test_memory_manager_delete_all_user_data(memory_manager):
    user_id = "delete-user-001"
    memory_manager.capture_user_statement(user_id, "hey my name is mayur", None)
    memory_manager.archive_session(
        user_id,
        [
            {"role": "user", "content": "Tell me about RAG"},
            {"role": "assistant", "content": "RAG uses retrieval and generation."},
        ],
    )
    result = memory_manager.delete_all_user_data(user_id)
    assert result["deleted"] is True
    profile = memory_manager.get_profile(user_id)
    assert profile["display_name"] == ""
    assert profile["facts"] == []
    assert profile["recent_sessions"] == []


def test_episodic_store_only_deletes_target_user(temp_episodic_store):
    emb = [0.1] * 384
    temp_episodic_store.add_memory("user a summary", emb, "user-a-123456", memory_id="mem-a")
    temp_episodic_store.add_memory("user b summary", emb, "user-b-123456", memory_id="mem-b")
    deleted = temp_episodic_store.delete_user_memories("user-a-123456")
    assert deleted == 1
    remaining = temp_episodic_store.collection.get(where={"user_id": "user-b-123456"}, include=[])
    assert len(remaining.get("ids") or []) == 1
