import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.txt_loader import TextDocumentLoader
from src.ingestion.pdf_loader import PDFDocumentLoader
from src.ingestion.html_loader import HTMLDocumentLoader


def test_text_loader(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("# Test Title\nThis is a sample document for testing.")
    
    loader = TextDocumentLoader()
    docs = loader.process(str(file))
    
    assert len(docs) == 1
    assert "sample document" in docs[0].content
    assert docs[0].metadata.title == "Test Title"


def test_html_loader(tmp_path):
    file = tmp_path / "page.html"
    file.write_text("<html><head><title>HTML Test Page</title></head><body><h1>Hello World</h1></body></html>")
    
    loader = HTMLDocumentLoader()
    docs = loader.process(str(file))
    
    assert len(docs) == 1
    assert "Hello World" in docs[0].content
    assert docs[0].metadata.title == "HTML Test Page"
