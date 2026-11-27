"""
Application Settings
Load environment variables from .env.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Project root: report-agent-project2/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

@dataclass(frozen=True)
class Settings:
    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------
    APP_NAME: str = "Report Writing Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_TIMEOUT_MS: int = int(
        os.getenv("DB_TIMEOUT_MS", "10000")
    )

    # --------------------------------------------------------
    # OpenAI LLM
    # --------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY", ""
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", ""
    )
    LLM_TIMEOUT_SECONDS: int = int(
        os.getenv("LLM_TIMEOUT_SECONDS", "30")
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", ""
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", ""
    )
    RERANK_BATCH_SIZE: int = int(
        os.getenv("RERANK_BATCH_SIZE", "2")
    )
    RERANK_MAX_LENGTH: int = int(
        os.getenv("RERANK_MAX_LENGTH", "512")
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------
    RETRIEVE_K: int = int(
        os.getenv("RETRIEVE_K", "8")
    )
    CONTEXT_K: int = int(
        os.getenv("CONTEXT_K", "3")
    )
    CONTEXT_CHARS: int = int(
        os.getenv("CONTEXT_CHARS", "9000")
    )

    # --------------------------------------------------------
    # Agent Limits
    # --------------------------------------------------------
    MAX_TOOL_CALLS: int = int(
        os.getenv("MAX_TOOL_CALLS", "4")
    )
    MAX_MODEL_CALLS: int = int(
        os.getenv("MAX_MODEL_CALLS", "14")
    )

    # --------------------------------------------------------
    # Evaluation / Test
    # --------------------------------------------------------
    DAY30_REPORT_ID: int = int(
        os.getenv("DAY30_REPORT_ID", "63")
    )
    DAY30_MISSING_ID: int = int(
        os.getenv("DAY30_MISSING_ID", "999999")
    )

    # --------------------------------------------------------
    # LangSmith
    # --------------------------------------------------------
    LANGSMITH_TRACING: bool = (
        os.getenv("LANGSMITH_TRACING", "false").lower()
        == "true"
    )
    LANGCHAIN_TRACING_V2: bool = (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
        == "true"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    def validate(self) -> None:
        required_settings = {
            "DATABASE_URL": self.DATABASE_URL,
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "LLM_MODEL": self.LLM_MODEL,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "RERANKER_MODEL": self.RERANKER_MODEL,
        }

        missing = [
            key
            for key, value in required_settings.items()
            if not value or not value.strip()
        ]

        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}"
            )

        positive_settings = {
            "DB_TIMEOUT_MS": self.DB_TIMEOUT_MS,
            "LLM_TIMEOUT_SECONDS": self.LLM_TIMEOUT_SECONDS,
            "RETRIEVE_K": self.RETRIEVE_K,
            "CONTEXT_K": self.CONTEXT_K,
            "CONTEXT_CHARS": self.CONTEXT_CHARS,
            "MAX_TOOL_CALLS": self.MAX_TOOL_CALLS,
            "MAX_MODEL_CALLS": self.MAX_MODEL_CALLS,
            "RERANK_BATCH_SIZE": self.RERANK_BATCH_SIZE,
            "RERANK_MAX_LENGTH": self.RERANK_MAX_LENGTH,
        }

        invalid = [
            key
            for key, value in positive_settings.items()
            if value <= 0
        ]

        if invalid:
            raise ValueError(
                f"Settings must be greater than zero: {invalid}"
            )
        if self.CONTEXT_K > self.RETRIEVE_K:
            raise ValueError(
                "CONTEXT_K cannot exceed RETRIEVE_K"
            )

# Shared settings instance
settings = Settings()