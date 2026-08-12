from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.runner import ScreeningRunner
from hirelens.config import Settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
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
