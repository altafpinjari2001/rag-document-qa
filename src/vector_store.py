"""
RAG Document Q&A System - Vector Store Module.

Manages ChromaDB vector store for document storage and retrieval.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain.schema import Document

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
    ):
        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create persist directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=embedding_manager.get_langchain_embeddings(),
        )

        logger.info(
            f"VectorStoreManager initialized "
            f"(collection={collection_name}, "
            f"persist_dir={persist_directory})"
        )

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Add documents to the vector store in batches.

        Args:
            documents: List of Document objects to store.
            batch_size: Number of documents per batch.

        Returns:
            List of document IDs.
        """
        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = self.vectorstore.add_documents(batch)
            all_ids.extend(ids)
            logger.info(
                f"Added batch {i // batch_size + 1} "
                f"({len(batch)} documents)"
            )

        logger.info(
            f"Total {len(all_ids)} documents added to vector store"
        )
        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Search query string.
            k: Number of results to return.
            score_threshold: Minimum similarity score filter.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        if score_threshold:
            results = [
                (doc, score)
                for doc, score in results
                if score >= score_threshold
            ]

        logger.info(
            f"Found {len(results)} results for query: "
            f"'{query[:50]}...'"
        )
        return results

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get a LangChain retriever from the vector store."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": 5},
        )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        collection = self.client.get_collection(self.collection_name)
        return collection.count()
