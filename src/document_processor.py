"""
RAG Document Q&A System - Document Processor Module.

Handles document loading, text extraction, and intelligent chunking.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents: loading, cleaning, and chunking."""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
        ".md": TextLoader,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,
        )
        logger.info(
            f"DocumentProcessor initialized (chunk_size={chunk_size}, "
            f"overlap={chunk_overlap})"
        )

    def load_document(self, file_path: str | Path) -> list[Document]:
        """
        Load a document from the given file path.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Document objects with page content and metadata.

        Raises:
            ValueError: If file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        loader_cls = self.LOADER_MAP[ext]
        loader = loader_cls(str(path))

        logger.info(f"Loading document: {path.name}")
        documents = loader.load()

        # Enrich metadata
        for doc in documents:
            doc.metadata.update({
                "source_file": path.name,
                "file_type": ext,
                "file_size_bytes": path.stat().st_size,
            })

        logger.info(
            f"Loaded {len(documents)} page(s) from {path.name}"
        )
        return documents

    def split_documents(
        self, documents: list[Document]
    ) -> list[Document]:
        """
        Split documents into smaller chunks for embedding.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(
            f"Split {len(documents)} document(s) into "
            f"{len(chunks)} chunks"
        )
        return chunks

    def process(self, file_path: str | Path) -> list[Document]:
        """
        End-to-end document processing: load → clean → chunk.

        Args:
            file_path: Path to the document file.

        Returns:
            List of processed and chunked Document objects.
        """
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        return chunks

    def process_multiple(
        self, file_paths: list[str | Path]
    ) -> list[Document]:
        """Process multiple documents and return combined chunks."""
        all_chunks = []
        for path in file_paths:
            try:
                chunks = self.process(path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue

        logger.info(
            f"Processed {len(file_paths)} files → "
            f"{len(all_chunks)} total chunks"
        )
        return all_chunks
