import os
from pathlib import Path
from typing import Literal
from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application Settings configured via environment variables or .env file.
    No hardcoded secrets or hardcoded paths.
    """
    APP_NAME: str = Field(default="Cost-Efficient RAG Service", description="Application Name")
    BASE_DIR: str = Field(default=str(ROOT_DIR), description="Base directory path")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    LOG_LEVEL: str = Field(default="INFO", description="Log verbosity level")

    # API Keys
    OPENAI_API_KEY: str = Field(default="mock-key-for-testing", description="OpenAI API Key")

    # Vector Store Config
    VECTOR_DB_TYPE: Literal["lancedb", "chromadb"] = Field(
        default="lancedb", description="Selected primary vector database"
    )
    LANCE_DB_PATH: str = Field(
        default=str(ROOT_DIR / "data" / "lancedb"), description="LanceDB storage directory"
    )
    CHROMA_DB_PATH: str = Field(
        default=str(ROOT_DIR / "data" / "chromadb"), description="ChromaDB storage directory"
    )

    # Embedding Config
    EMBEDDING_PROVIDER: Literal["openai", "mock", "sentence-transformers"] = Field(
        default="openai", description="Embedding provider"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    EMBEDDING_DIM: int = Field(default=1536, description="Embedding vector dimensionality")

    # Chunking Config
    CHUNK_SIZE: int = Field(default=500, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk overlap in characters")

    # Retrieval & LLM Config
    TOP_K: int = Field(default=5, description="Top-k chunks to retrieve")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="LLM model name for generation")
    LLM_TEMPERATURE: float = Field(default=0.0, description="LLM temperature setting")

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=str(ROOT_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore"
        )
    else:
        class Config:
            env_file = str(ROOT_DIR / ".env")
            env_file_encoding = "utf-8"
            extra = "ignore"


# Global settings instance singleton
settings = Settings()
