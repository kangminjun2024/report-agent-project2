"""
Application Runtime

Centralized runtime assembly.

Current stage:
- Settings
- PostgreSQL Engine
- CUDA Device
- Health Check Callbacks

Future stages:
- Embedding Model
- Retriever
- Reranker
- LLM
- LangChain Agent
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import torch
from sqlalchemy.engine import Engine
from app.db.connection import create_db_engine
from app.db.repository import ReportRepository
from app.core.settings import Settings, settings
from app.core.preflight import (
    check_database,
    check_gpu,
)

logger = logging.getLogger(__name__)

# ============================================================
# 1. Runtime Container
# ============================================================
@dataclass
class Runtime:

    settings: Settings
    db_engine: Engine
    device: torch.device
    repository: Any = None
    health_checks: dict[str, Callable] = field(
        default_factory=dict
    )
    # Future components
    embeddings: Any = None
    retriever: Any = None
    reranker: Any = None
    rag_pipeline: Any = None
    agent: Any = None
    # Advanced stage
    run_store: Any = None

    # --------------------------------------------------------
    # Health Check
    # --------------------------------------------------------
    def get_health_checks(self) -> dict:
        return {
            "database": lambda: check_database(
                self.db_engine
            ),
            "gpu": check_gpu,
        }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    def close(self) -> None:

        logger.info("Closing Runtime resources")
        self.db_engine.dispose()
        logger.info("Runtime resources closed")

# ============================================================
# 2. Runtime Builder
# ============================================================
def build_runtime(
    config: Settings = settings,
) -> Runtime:

    logger.info("Building application runtime")
    # 1. Validate settings
    config.validate()

    # 2. Create PostgreSQL Engine
    engine = create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=max(
            1,
            config.DB_TIMEOUT_MS // 1000,
        ),
        connect_args={
            "connect_timeout": max(
                1,
                config.DB_TIMEOUT_MS // 1000,
            )
        },
    )

    # 3. Select CUDA device
    if not torch.cuda.is_available():
        engine.dispose()
        raise RuntimeError(
            "CUDA is not available. "
            "Please check the PyTorch GPU installation."
        )

    device = torch.device("cuda:0")

    logger.info(
        "GPU selected: %s",
        torch.cuda.get_device_name(0),
    )

    # 4. Assemble Runtime
    runtime = Runtime(
        settings=config,
        db_engine=engine,
        device=device,
    )

    runtime.health_checks = runtime.get_health_checks()
    logger.info("Runtime successfully created")

    return runtime
