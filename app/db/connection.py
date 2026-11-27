
"""
Database Connection Manager

Responsibilities:
- PostgreSQL Engine creation
- Connection pooling
- Read-only transaction
- Statement timeout
- Safe database resource management
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from app.core.settings import Settings, settings

logger = logging.getLogger(__name__)

# ============================================================
# 1. Database Engine Factory
# ============================================================
def create_db_engine(
    config: Settings = settings,
) -> Engine:
    """
    Create a PostgreSQL Engine.
    PostgreSQL connections are configured as read-only.
    """
    if not config.DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured")

    timeout_seconds = max(
        1,
        config.DB_TIMEOUT_MS // 1000,
    )

    engine = create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=timeout_seconds,
        pool_recycle=1800,
        connect_args={
            "connect_timeout": timeout_seconds,
            "options": (
                "-c default_transaction_read_only=on "
                f"-c statement_timeout={config.DB_TIMEOUT_MS}"
            ),
        },
        echo=False,
    )
    logger.info("PostgreSQL Engine created")

    return engine

# ============================================================
# 2. Read-only Transaction
# ============================================================
@contextmanager
def readonly_transaction(
    engine: Engine,
) -> Generator[Connection, None, None]:
    """
    Provide a read-only database transaction.
    Usage:
        with readonly_transaction(engine) as connection:
            result = connection.execute(
                text("SELECT 1")
            )

    The transaction is automatically committed or rolled back
    by SQLAlchemy. No write operations are permitted.
    """
    connection = engine.connect()

    try:
        with connection.begin():
            # Explicit transaction-level read-only protection.
            connection.execute(
                text("SET TRANSACTION READ ONLY")
            )

            # Apply timeout to the current transaction.
            connection.execute(
                text(
                    "SELECT set_config("
                    "'statement_timeout', :timeout, true)"
                ),
                {
                    "timeout": "10000",
                },
            )
            yield connection

    except Exception:
        logger.exception(
            "Read-only database transaction failed"
        )
        raise
    finally:
        connection.close()

# ============================================================
# 3. Database Ping
# ============================================================
def ping_database(
    engine: Engine,
) -> dict:
    """
    Check PostgreSQL connectivity using SELECT 1.
    """
    try:
        with readonly_transaction(engine) as connection:

            result = connection.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        return {
            "status": "healthy",
            "database": "postgresql",
            "result": value,
        }

    except Exception as exc:
        logger.exception("Database ping failed")
        return {
            "status": "unhealthy",
            "error": type(exc).__name__,
        }

# ============================================================
# 4. Database Shutdown
# ============================================================
def close_db_engine(
    engine: Engine,
) -> None:
    """
    Dispose database connection pool.
    """
    if engine is not None:
        engine.dispose()
        logger.info("PostgreSQL Engine disposed")
