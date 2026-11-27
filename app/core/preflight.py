
"""
Preflight Checks

Check infrastructure before Agent execution.
"""

import logging
from typing import Any

import torch
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.settings import Settings, settings


logger = logging.getLogger(__name__)


# ============================================================
# 1. Settings Check
# ============================================================

def check_settings(
    config: Settings = settings,
) -> dict:

    try:
        config.validate()

        return {
            "status": "healthy",
            "message": "Settings validation successful",
        }

    except Exception as exc:

        logger.exception("Settings check failed")

        return {
            "status": "unhealthy",
            "error": str(exc),
        }


# ============================================================
# 2. Database Check
# ============================================================

def check_database(
    engine: Engine,
) -> dict:

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT 1")
            )

            value = result.scalar_one()

        if value != 1:
            raise RuntimeError(
                "Unexpected database response"
            )

        return {
            "status": "healthy",
            "database": "postgresql",
            "query": "SELECT 1",
        }

    except Exception as exc:

        logger.exception("Database health check failed")

        return {
            "status": "unhealthy",
            "error": type(exc).__name__,
        }


# ============================================================
# 3. GPU / CUDA Check
# ============================================================

def check_gpu() -> dict:

    try:

        cuda_available = torch.cuda.is_available()

        if not cuda_available:

            return {
                "status": "unhealthy",
                "cuda_available": False,
                "message": "CUDA is not available",
            }

        device_count = torch.cuda.device_count()

        device_name = torch.cuda.get_device_name(0)

        return {
            "status": "healthy",
            "cuda_available": True,
            "device_name": device_name,
            "device_count": device_count,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    except Exception as exc:

        logger.exception("GPU health check failed")

        return {
            "status": "unhealthy",
            "cuda_available": False,
            "error": type(exc).__name__,
        }


# ============================================================
# 4. Full Preflight
# ============================================================

def run_preflight(
    engine: Engine,
    config: Settings = settings,
) -> dict:

    results = {
        "settings": check_settings(config),
        "database": check_database(engine),
        "gpu": check_gpu(),
    }

    statuses = [
        result["status"]
        for result in results.values()
    ]

    overall_status = (
        "healthy"
        if all(status == "healthy" for status in statuses)
        else "unhealthy"
    )

    return {
        "status": overall_status,
        "checks": results,
    }
