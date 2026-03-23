"""
RAG Document Q&A System - Configuration Module.

Centralized configuration management using pydantic-settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # LLM Configuration
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", env="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")

    # Chunking Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # Retrieval Configuration
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    use_hybrid_search: bool = Field(default=True, env="USE_HYBRID_SEARCH")

    # Vector Store
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", env="CHROMA_PERSIST_DIR"
    )
    collection_name: str = Field(default="documents", env="COLLECTION_NAME")

    # Application
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    return Settings()


# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
