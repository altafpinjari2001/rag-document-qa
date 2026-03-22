"""
RAG Document Q&A System - Pipeline Module.

End-to-end RAG pipeline orchestrating all components.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain.schema import Document

from .config import get_settings
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retriever import HybridRetriever
from .generator import ResponseGenerator, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Orchestrates document processing, embedding, storage,
    retrieval, and response generation.

    Usage:
        >>> rag = RAGPipeline()
        >>> rag.ingest("document.pdf")
        >>> response = rag.query("What is the main topic?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        use_hybrid_search: bool = True,
        persist_directory: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            api_key=self.api_key,
        )

        self.vector_store = VectorStoreManager(
            embedding_manager=self.embedding_manager,
            persist_directory=(
                persist_directory or settings.chroma_persist_dir
            ),
        )

        self.generator = ResponseGenerator(
            model_name=llm_model,
            api_key=self.api_key,
            streaming=True,
        )

        self.top_k = top_k
        self.use_hybrid = use_hybrid_search
        self._all_chunks: list[Document] = []
        self._retriever: Optional[HybridRetriever] = None

        logger.info("RAGPipeline initialized successfully")

    def ingest(
        self, file_path: str | Path
    ) -> dict[str, int]:
        """
        Ingest a document into the RAG pipeline.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary with ingestion statistics.
        """
        # Process document
        chunks = self.doc_processor.process(file_path)
        self._all_chunks.extend(chunks)

        # Add to vector store
        ids = self.vector_store.add_documents(chunks)

        # Rebuild retriever with updated documents
        self._retriever = HybridRetriever(
            vector_store=self.vector_store,
            documents=self._all_chunks,
            top_k=self.top_k,
            use_hybrid=self.use_hybrid,
        )

        stats = {
            "file": str(file_path),
            "chunks_created": len(chunks),
            "documents_stored": len(ids),
            "total_documents": self.vector_store.count,
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def ingest_multiple(
        self, file_paths: list[str | Path]
    ) -> dict[str, int]:
        """Ingest multiple documents."""
        total_chunks = 0
        for path in file_paths:
            stats = self.ingest(path)
            total_chunks += stats["chunks_created"]

        return {
            "files_processed": len(file_paths),
            "total_chunks": total_chunks,
            "total_documents": self.vector_store.count,
        }

    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG pipeline with a question.

        Args:
            question: User's natural language question.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        if not self._retriever:
            return RAGResponse(
                answer="No documents have been ingested yet. "
                "Please upload documents first."
            )

        # Retrieve relevant chunks
        retrieved_docs = self._retriever.retrieve(question)

        # Generate response
        response = self.generator.generate(
            question=question,
            retrieved_documents=retrieved_docs,
        )

        logger.info(
            f"Query answered. Sources: {len(response.sources)}"
        )
        return response

    def clear(self) -> None:
        """Clear all stored documents and reset pipeline."""
        self.vector_store.delete_collection()
        self._all_chunks.clear()
        self._retriever = None
        logger.info("Pipeline cleared")
