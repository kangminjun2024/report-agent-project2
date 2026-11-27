
"""
FastAPI Application Entry Point
Project: Report Agent

Responsibilities:
- FastAPI application initialization
- Application lifespan management
- API router registration
- Health check (Server / Database / GPU)
- Common exception handling

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""


import torch
from sqlalchemy import create_engine, text
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib import import_module
import inspect
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


# ============================================================
# 1. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("report-agent")


# ============================================================
# 2. Application Configuration
# ============================================================

APP_NAME = "Report Writing Agent"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"


# ============================================================
# 3. Router Registration
# ============================================================

def register_router(
    app: FastAPI,
    module_path: str,
    router_name: str = "router",
) -> None:
    """
    Import and register an API router.

    The router module can be registered after its implementation
    is completed.

    If the module exists but does not expose a router yet,
    registration is skipped with a warning.
    """

    try:
        module = import_module(module_path)

        router = getattr(module, router_name, None)

        if router is None:
            logger.warning(
                "Router not registered: %s.%s is not defined",
                module_path,
                router_name,
            )
            return

        app.include_router(router)

        logger.info("Router registered: %s", module_path)

    except ModuleNotFoundError as exc:
        # Do not hide missing dependencies inside an existing module.
        if exc.name != module_path:
            raise

        logger.warning(
            "Router module not available yet: %s",
            module_path,
        )


# ============================================================
# 4. Infrastructure Health Check
# ============================================================

def check_database():
    """
    Temporary PostgreSQL health check.

    DATABASE_URL example:
    postgresql+psycopg://user:password@localhost:5432/dbname
    """

    from app.core.settings import settings

    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "postgresql",
        }

    finally:
        engine.dispose()


def check_gpu():
    """
    Check PyTorch CUDA availability.
    """

    if not torch.cuda.is_available():
        return {
            "status": "unhealthy",
            "cuda_available": False,
            "message": "CUDA is not available.",
        }

    return {
        "status": "healthy",
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }

# ============================================================
# 4. Health Check Helpers
# ============================================================

async def execute_health_check(
    name: str,
    checker: Any,
) -> dict:
    """
    Execute a synchronous or asynchronous health checker.

    Each checker should return:
        True  -> healthy
        False -> unhealthy
        dict  -> detailed health information
    """

    if checker is None:
        return {
            "status": "not_configured",
        }

    try:
        result = checker()

        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, dict):
            return {
                **result,
                "status": result.get("status", "unknown"),
            }

        return {
            "status": "healthy" if result is True else "unhealthy",
        }

    except Exception as exc:
        logger.exception(
            "Health check failed: %s",
            name,
        )

        return {
            "status": "unhealthy",
            "error": type(exc).__name__,
        }


async def collect_health_status(app: FastAPI) -> dict:
    """
    Collect server, database and GPU readiness information.

    Health check functions will be injected through:
        app.state.health_checks
    """

    checks = {
        "server": {
            "status": "healthy",
        },
        "database": await execute_health_check(
            "database",
            app.state.health_checks.get("database"),
        ),
        "gpu": await execute_health_check(
            "gpu",
            app.state.health_checks.get("gpu"),
        ),
    }

    statuses = [
        item["status"]
        for item in checks.values()
    ]

    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "not_configured" in statuses:
        overall_status = "not_ready"
    elif "unknown" in statuses:
        overall_status = "unknown"
    else:
        overall_status = "healthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


# ============================================================
# 5. Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Heavy model initialization and database preparation
    will eventually be delegated to core/runtime.py.

    Do not load GPU models directly in this file.
    """

    logger.info("=" * 60)
    logger.info("Starting %s", APP_NAME)
    logger.info("Version: %s", APP_VERSION)
    logger.info("=" * 60)

    # Shared application state
    app.state.started_at = time.time()

    # Health check callbacks.
    # These will be connected to preflight.py/runtime.py later.
    app.state.health_checks = {
        "database": check_database,
        "gpu": check_gpu,
    }

    # Runtime and run store will be injected later.
    app.state.runtime = None
    app.state.run_store = None

    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down %s", APP_NAME)

    runtime = getattr(app.state, "runtime", None)

    if runtime is not None:
        close_method = getattr(runtime, "aclose", None)

        if close_method is not None:
            result = close_method()

            if inspect.isawaitable(result):
                await result

    logger.info("Application shutdown completed")


# ============================================================
# 6. FastAPI Application
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "LangChain-based Report Writing Agent API "
        "with RAG, Evidence, Trace and Structured Output."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================
# 7. Common Exception Handlers
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters.",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
            },
        },
    )


# ============================================================
# 8. API-00: Health Check
# ============================================================

@app.get(
    "/health",
    tags=["Health"],
    summary="Server, Database and GPU Health Check",
)
async def health_check():
    """
    Check basic server and infrastructure readiness.

    HTTP 200: All configured checks are healthy.
    HTTP 503: One or more checks are unhealthy,
              or required checks are not configured yet.
    """

    result = await collect_health_status(app)

    http_status = (
        200
        if result["status"] == "healthy"
        else 503
    )

    return JSONResponse(
        status_code=http_status,
        content=result,
    )


# ============================================================
# 9. API Routers
# ============================================================

# API-01: POST /api/v1/agent/ask
register_router(
    app,
    "app.api.agent",
)

# API-02: POST /api/v1/rag/debug
register_router(
    app,
    "app.api.rag",
)


# ============================================================
# 10. Root Endpoint
# ============================================================

@app.get(
    "/",
    tags=["Root"],
    summary="Application Information",
)
async def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# 11. Application Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
