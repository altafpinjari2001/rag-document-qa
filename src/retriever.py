"""
RAG Document Q&A System - Retriever Module.

Implements hybrid retrieval combining semantic search with BM25 keyword matching.
"""

import logging
from typing import Optional

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining semantic search (dense) with
    BM25 keyword matching (sparse) for improved retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        documents: Optional[list[Document]] = None,
        top_k: int = 5,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_hybrid: bool = True,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.use_hybrid = use_hybrid

        # Semantic retriever (dense)
        self.semantic_retriever = vector_store.get_retriever(
            search_kwargs={"k": top_k}
        )

        # BM25 retriever (sparse) — if hybrid mode is enabled
        self.bm25_retriever = None
        self.ensemble_retriever = None

        if use_hybrid and documents:
            self._setup_hybrid(
                documents, semantic_weight, bm25_weight
            )

        logger.info(
            f"HybridRetriever initialized "
            f"(hybrid={use_hybrid}, top_k={top_k})"
        )

    def _setup_hybrid(
        self,
        documents: list[Document],
        semantic_weight: float,
        bm25_weight: float,
    ) -> None:
        """Initialize BM25 retriever and ensemble."""
        self.bm25_retriever = BM25Retriever.from_documents(
            documents, k=self.top_k
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.semantic_retriever,
                self.bm25_retriever,
            ],
            weights=[semantic_weight, bm25_weight],
        )
        logger.info(
            f"Hybrid search enabled "
            f"(semantic={semantic_weight}, bm25={bm25_weight})"
        )

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for the given query.

        Uses hybrid (ensemble) retrieval if available,
        otherwise falls back to semantic-only retrieval.

        Args:
            query: User's question or search query.

        Returns:
            List of relevant Document objects with metadata.
        """
        if self.use_hybrid and self.ensemble_retriever:
            results = self.ensemble_retriever.invoke(query)
        else:
            results = self.semantic_retriever.invoke(query)

        logger.info(
            f"Retrieved {len(results)} documents for: "
            f"'{query[:50]}...'"
        )
        return results[: self.top_k]

    def retrieve_with_scores(
        self, query: str
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        return self.vector_store.similarity_search(
            query, k=self.top_k
        )
