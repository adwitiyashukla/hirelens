"""FastAPI application factory.

A factory rather than a module-level ``app`` object, because tests need an
application wired to an in-memory database and a fake LLM provider. With a global
app that requires monkeypatching; with a factory it is an argument.

Shared state (engine, session factory, background runner) lives on ``app.state``
and is created in the lifespan handler, so it is built once at startup and torn
down deterministically at shutdown rather than leaking connections between tests.
"""

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
    """Build the application.

    Every dependency is injectable, which is what makes the route tests fast and
    hermetic: an in-memory database, a runner backed by a fake provider, no network.
    """
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
            # Cancel in-flight runs before disposing the engine, or their final
            # writes hit a closed pool and log alarming errors on every shutdown.
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
        # The frontend is served from a different origin in development. Locked
        # down by configuration in a real deployment.
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
        """Domain validation errors are client errors, not server errors.

        The pipeline raises ValueError for things like a job description too short
        to compile. Without this they would surface as 500s and look like bugs.
        """
        return JSONResponse(status_code=400, content={"detail": str(exc), "code": "invalid_input"})

    # Mounted last. A mount at "/" matches every path the routers above did not
    # claim, so registering it earlier would shadow the entire API.
    _mount_frontend(app)

    return app


def find_static_dir() -> Path | None:
    """Locate the built frontend, if there is one.

    Returns ``None`` when the dashboard has not been built, which is the normal
    state in development (Vite serves it on port 5173 and proxies the API) and
    during tests. The API is fully usable without it, so a missing build is not
    an error.
    """
    override = os.getenv("HIRELENS_STATIC_DIR")
    if override:
        path = Path(override)
        return path if (path / "index.html").is_file() else None

    # From src/hirelens/api/app.py up to the repository root, then into web/dist.
    candidate = Path(__file__).resolve().parents[3] / "web" / "dist"
    return candidate if (candidate / "index.html").is_file() else None


def _mount_frontend(app: FastAPI) -> None:
    """Serve the dashboard from the API when a build is present.

    One origin for both means no CORS in production and one process to deploy,
    which matters when the target is a free hosting tier that gives you exactly
    one container. ``html=True`` makes StaticFiles fall back to ``index.html``,
    so a refresh on any path loads the app rather than returning 404.
    """
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
        """Liveness plus configuration.

        Reports whether a provider credential is present without ever returning
        it, so a deployment can be diagnosed from the outside.
        """
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


# Module-level app for `uvicorn hirelens.api.app:app`.
app = create_app()
