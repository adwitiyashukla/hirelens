from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from hirelens import __version__
from hirelens.api.db.session import (
    DEFAULT_DATABASE_URL,
    create_engine,
    create_schema,
    create_session_factory,
    ping,
)
from hirelens.api.routes import documents, jobs, runs
from hirelens.api.runner import ScreeningRunner
from hirelens.api.schemas import HealthOut
from hirelens.config import Settings, get_settings

logger = logging.getLogger(__name__)

DESCRIPTION = """
Evidence-grounded candidate screening.

Every score returned by this API cites the exact character range of the resume it
came from, carries a confidence band from repeated sampling, and is produced with
identifying details masked by default.

**Typical flow**

1. `POST /api/jobs` with the job description. It compiles into a weighted rubric.
2. `POST /api/documents` with one or more resumes. Uploads are idempotent: the
   document id is the hash of the file bytes.
3. `POST /api/runs` to screen those documents against that job. Returns immediately.
4. `GET /api/runs/{id}/events` to stream progress, or poll `GET /api/runs/{id}`.
5. `GET /api/runs/{id}/shortlist` for the ranked list, then
   `GET /api/assessments/{id}` for one candidate with citations and highlight boxes.
"""


def create_app(
    *,
    database_url: str | None = None,
    settings: Settings | None = None,
    runner: ScreeningRunner | None = None,
    create_tables: bool = True,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    url = database_url or os.getenv("HIRELENS_DATABASE_URL", DEFAULT_DATABASE_URL)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(url)
        session_factory = create_session_factory(engine)

        if create_tables:
            await create_schema(engine)

        application.state.engine = engine
        application.state.session_factory = session_factory
        application.state.settings = resolved_settings
        application.state.runner = runner or ScreeningRunner(
            session_factory, settings=resolved_settings
        )

        logger.info("hirelens api ready (database=%s)", url.split("://")[0])
        try:
            yield
        finally:
            await application.state.runner.shutdown()
            await engine.dispose()

    app = FastAPI(
        title="HireLens",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(_health_router(resolved_settings))
    app.include_router(jobs.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc), "code": "invalid_input"})

    _mount_frontend(app)

    return app


def find_static_dir() -> Path | None:
    override = os.getenv("HIRELENS_STATIC_DIR")
    if override:
        path = Path(override)
        return path if (path / "index.html").is_file() else None

    candidate = Path(__file__).resolve().parents[3] / "web" / "dist"
    return candidate if (candidate / "index.html").is_file() else None


def _mount_frontend(app: FastAPI) -> None:
    static_dir = find_static_dir()
    if static_dir is None:
        logger.info("no frontend build found; serving the API only")
        return

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    logger.info("serving dashboard from %s", static_dir)


def _health_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=HealthOut)
    async def health(request: Request) -> HealthOut:
        database_ok = await ping(request.app.state.engine)
        return HealthOut(
            status="ok" if database_ok else "degraded",
            version=__version__,
            database=database_ok,
            provider=str(settings.llm_provider),
            model=settings.active_model,
            provider_configured=settings.has_credentials,
            blind_mode=settings.blind_mode,
        )

    return router


app = create_app()
