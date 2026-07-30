"""Engine and session management.

SQLite is the default and Postgres is a URL change. That is not a compromise: the
whole point of using SQLAlchemy rather than raw SQL is that the same code runs on
both, and defaulting to SQLite means `uvicorn hirelens.api.app:app` works on a
fresh clone with no Docker, no server, and no connection string. Tests run against
in-memory SQLite and finish in milliseconds. Deployment points ``HIRELENS_DATABASE_URL``
at Postgres and nothing else changes.

Two SQLite-specific details are handled here because getting them wrong produces
bugs that only appear under concurrency, which is exactly when they are hardest to
debug:

* **WAL mode**, so a read does not block while the background runner writes.
* **A shared static pool for in-memory databases**, because ``sqlite+aiosqlite:///:memory:``
  otherwise gives every connection its own private database, and a test that
  writes on one connection then reads on another sees an empty schema.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from hirelens.api.db.models import Base

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./hirelens.db"


def is_memory_url(url: str) -> bool:
    return ":memory:" in url


def create_engine(url: str = DEFAULT_DATABASE_URL, *, echo: bool = False) -> AsyncEngine:
    """Build an async engine appropriate to the backend named in ``url``."""
    kwargs: dict[str, object] = {"echo": echo, "future": True}

    if url.startswith("sqlite"):
        # SQLite rejects cross-thread use by default; the async driver hands
        # connections between threads legitimately.
        kwargs["connect_args"] = {"check_same_thread": False}
        if is_memory_url(url):
            # One connection shared by everyone, or each caller silently gets its
            # own empty database.
            kwargs["poolclass"] = StaticPool
    else:
        # Postgres: recycle before typical proxy idle timeouts, and check
        # liveness rather than handing out a dead connection.
        kwargs["pool_pre_ping"] = True
        kwargs["pool_recycle"] = 1800

    engine = create_async_engine(url, **kwargs)

    if url.startswith("sqlite") and not is_memory_url(url):
        _enable_sqlite_concurrency(engine)

    return engine


def _enable_sqlite_concurrency(engine: AsyncEngine) -> None:
    """Turn on WAL and foreign keys for file-backed SQLite.

    Without WAL, the background screening runner holds a write lock and every API
    read blocks behind it, which looks exactly like the server hanging. Foreign
    keys are off by default in SQLite, so the cascade deletes declared in the
    models would silently not happen.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        # Objects stay usable after commit. Without this, returning an ORM object
        # from a route triggers a lazy refresh on a closed session, which fails
        # with a confusing DetachedInstanceError.
        expire_on_commit=False,
        autoflush=False,
    )


async def create_schema(engine: AsyncEngine) -> None:
    """Create tables if absent.

    Fine for SQLite and for a demo deployment. A real Postgres rollout would use
    Alembic migrations; this is the honest shortcut and it is labelled as one.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def drop_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


async def ping(engine: AsyncEngine) -> bool:
    """True when the database answers. Used by the health endpoint."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("database ping failed: %s", exc)
        return False


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A session that commits on success and rolls back on failure.

    Used by the background runner, which has no request to hang a session off.
    """
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
