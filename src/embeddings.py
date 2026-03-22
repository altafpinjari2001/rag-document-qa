"""
RAG Document Q&A System - Embeddings Module.

Wrapper around embedding models for consistent interface.
"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model initialization and operations."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: int = 1536,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,
            dimensions=dimensions,
        )
        logger.info(
            f"EmbeddingManager initialized with model={model_name}, "
            f"dimensions={dimensions}"
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query string."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts."""
        logger.info(f"Embedding {len(texts)} documents...")
        return self.embeddings.embed_documents(texts)

    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self.embeddings
