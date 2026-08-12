from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from hirelens.api.db.repository import JobRepository, RunRepository
from hirelens.api.deps import SessionDep
from hirelens.api.schemas import JobCreate, JobOut, RequirementOut, RunOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


def to_out(job) -> JobOut:
    requirements = []
    if job.rubric_json:
        requirements = [
            RequirementOut(**requirement) for requirement in job.rubric_json.get("requirements", [])
        ]
    return JobOut(
        id=job.id,
        title=job.title,
        description=job.description,
        rubric_id=job.rubric_id,
        created_at=job.created_at,
        requirements=requirements,
    )


@router.post("", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def create_job(payload: JobCreate, session: AsyncSession = SessionDep) -> JobOut:
    job = await JobRepository(session).upsert(payload.description, title=payload.title)
    return to_out(job)


@router.get("", response_model=list[JobOut])
async def list_jobs(
    limit: int = 50, offset: int = 0, session: AsyncSession = SessionDep
) -> list[JobOut]:
    jobs = await JobRepository(session).list(limit=min(limit, 200), offset=offset)
    return [to_out(job) for job in jobs]


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str, session: AsyncSession = SessionDep) -> JobOut:
    job = await JobRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return to_out(job)


@router.get("/{job_id}/runs", response_model=list[RunOut])
async def list_job_runs(
    job_id: str, limit: int = 20, session: AsyncSession = SessionDep
) -> list[RunOut]:
    job = await JobRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    runs = await RunRepository(session).list_for_job(job_id, limit=min(limit, 100))
    return [RunOut.model_validate(run) for run in runs]
