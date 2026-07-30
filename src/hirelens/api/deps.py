"""Shared FastAPI dependencies.

Everything is pulled off ``app.state`` rather than from module globals, so a test
application with an in-memory database and a fake provider needs no patching.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.runner import ScreeningRunner
from hirelens.config import Settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """A request-scoped session that commits on success, rolls back on failure.

    Committing here rather than in each route means a handler that raises halfway
    through cannot leave a half-written run behind.
    """
    factory = request.app.state.session_factory
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_runner(request: Request) -> ScreeningRunner:
    return request.app.state.runner


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


SessionDep = Depends(get_session)
RunnerDep = Depends(get_runner)
SettingsDep = Depends(get_settings_dep)
