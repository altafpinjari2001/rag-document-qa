"""Tests for the DocumentProcessor module."""

import tempfile
from pathlib import Path

import pytest

from src.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Create a DocumentProcessor with default settings."""
    return DocumentProcessor(chunk_size=500, chunk_overlap=100)


@pytest.fixture
def sample_txt_file():
    """Create a temporary text file for testing."""
    content = (
        "This is a sample document for testing the RAG pipeline. "
        "It contains multiple sentences to ensure proper chunking. "
        "The document processor should be able to load and split "
        "this text into meaningful chunks.\n\n"
        "This is a second paragraph with different content. "
        "It discusses the importance of retrieval-augmented "
        "generation in modern AI systems. RAG combines the power "
        "of large language models with external knowledge sources.\n\n"
        "The third paragraph covers vector databases and embeddings. "
        "ChromaDB is a popular choice for storing document embeddings "
        "and performing similarity search operations efficiently."
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write(content)
        return f.name


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""

    def test_load_txt_document(self, processor, sample_txt_file):
        """Test loading a text document."""
        docs = processor.load_document(sample_txt_file)
        assert len(docs) > 0
        assert docs[0].page_content != ""
        assert "source_file" in docs[0].metadata

    def test_split_documents(self, processor, sample_txt_file):
        """Test document splitting into chunks."""
        docs = processor.load_document(sample_txt_file)
        chunks = processor.split_documents(docs)
        assert len(chunks) >= 1
        assert all(
            "chunk_index" in c.metadata for c in chunks
        )

    def test_process_end_to_end(self, processor, sample_txt_file):
        """Test end-to-end processing."""
        chunks = processor.process(sample_txt_file)
        assert len(chunks) >= 1
        assert chunks[0].metadata["total_chunks"] == len(chunks)

    def test_unsupported_file_raises_error(self, processor):
        """Test that unsupported file types raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            with pytest.raises(ValueError, match="Unsupported"):
                processor.load_document(f.name)

    def test_missing_file_raises_error(self, processor):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            processor.load_document("/nonexistent/file.txt")
