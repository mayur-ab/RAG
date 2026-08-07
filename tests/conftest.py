"""Pytest configuration — force mock providers so tests run without Ollama/GPU."""
import os

os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["RERANKER_PROVIDER"] = "identity"
os.environ["VECTOR_STORE_PROVIDER"] = "chroma"
os.environ["VECTOR_DB_PATH"] = "./test_chroma_db"
os.environ["USER_MEMORY_DB_PATH"] = "./test_user_memory.db"
os.environ["USER_MEMORY_CHROMA_PATH"] = "./test_user_memory_chroma"
os.environ["DOC_CLASSIFICATION_DB_PATH"] = "./test_document_metadata.db"
os.environ["ENABLE_DOC_CLASSIFICATION"] = "true"
os.environ["ENABLE_USER_MEMORY"] = "true"
