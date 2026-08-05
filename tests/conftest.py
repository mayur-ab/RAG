"""Pytest configuration — force mock providers so tests run without Ollama/GPU."""
import os

os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["RERANKER_PROVIDER"] = "identity"
os.environ["VECTOR_STORE_PROVIDER"] = "chroma"
os.environ["VECTOR_DB_PATH"] = "./test_chroma_db"
