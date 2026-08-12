from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from hirelens.api.app import create_app, find_static_dir
from hirelens.api.runner import ProgressChannel, ScreeningRunner
from hirelens.api.schemas import RunProgress
from hirelens.assess.pipeline import ScreeningPipeline
from hirelens.config import Provider, Settings
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve.embeddings import HashingEmbedder

RESUME = """PRIYA RAMAN
priya.raman@example.com | Bengaluru

EXPERIENCE
Backend Engineer, Fintech Co. (2022 - present)
Cut p99 checkout latency from 1.2s to 180ms in the payments path.
Deployed and operated the reconciliation service on Kubernetes.

PROJECTS
kvstore: Raft-based distributed key-value store in Go

EDUCATION
B.Tech Computer Science, State University (2022)

SKILLS
Go, Python, PostgreSQL, Kubernetes
"""

JOB_DESCRIPTION = """
Senior Backend Engineer, Payments Platform

You will own the services behind our payments platform, design and operate systems
processing millions of events per day, and carry the pager for what you build.

Requirements
- Strong experience running containerised workloads in production
- A track record of measurably improving system performance
"""


class StubProvider(LLMProvider):
    name = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        convo = "\n".join(message.content for message in request.messages)
        self.calls += 1

        if "Compile the following job description" in convo:
            payload: dict = {
                "role_title": "Senior Backend Engineer",
                "seniority": "senior",
                "requirements": [
                    {
                        "text": "Has run containerised workloads in production",
                        "kind": "must_have",
                        "category": "experience",
                        "evidence_hint": "Kubernetes deployed operated production",
                    },
                    {
                        "text": "Has measurably improved system performance",
                        "kind": "must_have",
                        "category": "experience",
                        "evidence_hint": "latency p99 reduced milliseconds",
                    },
                ],
            }
        elif "Extract paid professional experience" in convo:
            payload = {
                "work": [
                    {
                        "company": {"value": "Fintech Co.", "quote": "Fintech Co."},
                        "position": {"value": "Backend Engineer", "quote": "Backend Engineer"},
                        "highlights": [
                            {
                                "value": "Cut p99 latency to 180ms",
                                "quote": "Cut p99 checkout latency from 1.2s to 180ms in the payments path.",
                            },
                            {
                                "value": "Operated on Kubernetes",
                                "quote": "Deployed and operated the reconciliation service on Kubernetes.",
                            },
                        ],
                    }
                ]
            }
        elif "REQUIREMENT:" in convo:
            evidence = convo.split("EVIDENCE RETRIEVED")[1]
            verdict = "clear" if ("Kubernetes" in evidence or "p99" in evidence) else "none"
            ids = [line.split("]")[0][1:] for line in convo.split("\n") if line.startswith("[")][:1]
            payload = {
                "verdict": verdict,
                "reasoning": "stubbed judgement",
                "evidence_unit_ids": ids if verdict != "none" else [],
            }
        elif "Uncertainties to resolve" in convo:
            payload = {
                "questions": [
                    {
                        "question": "How did you verify the latency improvement?",
                        "rationale": "Establishes rigour behind the headline metric.",
                        "targets": "performance",
                    }
                ]
            }
        else:
            payload = {}

        return CompletionResponse(
            content=json.dumps(payload), model=self.model, usage=Usage(80, 25)
        )

    async def aclose(self) -> None:
        return None


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        llm_provider=Provider.OLLAMA,
        cache_enabled=False,
        blind_mode=True,
        self_consistency_k=1,
        requests_per_minute=0,
    )

    app = create_app(database_url="sqlite+aiosqlite:///:memory:", settings=settings)

    def pipeline_factory() -> ScreeningPipeline:
        return ScreeningPipeline(
            LLMClient(StubProvider(), settings=settings),
            settings=settings,
            embedder=HashingEmbedder(),
        )

    async with (
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        app.state.runner = ScreeningRunner(
            app.state.session_factory,
            settings=settings,
            pipeline_factory=pipeline_factory,
        )
        yield http_client


async def upload_resume(client: httpx.AsyncClient, name: str = "priya.txt") -> str:
    response = await client.post(
        "/api/documents", files={"files": (name, RESUME.encode(), "text/plain")}
    )
    assert response.status_code == 200, response.text
    return response.json()["uploaded"][0]["document"]["id"]


async def create_job(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/jobs", json={"description": JOB_DESCRIPTION})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def run_to_completion(
    client: httpx.AsyncClient, job_id: str, document_ids: list[str], **options
) -> str:
    response = await client.post(
        "/api/runs", json={"job_id": job_id, "document_ids": document_ids, **options}
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]

    deadline = time.monotonic() + 60.0
    status: dict[str, object] = {}

    while time.monotonic() < deadline:
        status = (await client.get(f"/api/runs/{run_id}")).json()
        if status["status"] in ("completed", "failed"):
            assert status["status"] == "completed", status.get("error")
            return run_id
        await asyncio.sleep(0.05)

    raise AssertionError(
        f"run {run_id} did not reach a terminal state within 60s. "
        f"Last seen: status={status.get('status')!r} stage={status.get('stage')!r} "
        f"completed={status.get('completed')} failed={status.get('failed')} "
        f"of {status.get('total')}"
    )


class TestHealth:
    async def test_reports_ok(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["database"] is True

    async def test_reports_configuration_without_leaking_the_key(
        self, client: httpx.AsyncClient
    ) -> None:
        body = response_json = (await client.get("/health")).json()
        assert "provider" in body
        assert body["blind_mode"] is True
        assert not any("key" in str(value).lower() for value in response_json.values())

    async def test_openapi_is_served(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/openapi.json")).status_code == 200


class TestJobs:
    async def test_create_and_fetch(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        response = await client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    async def test_creation_is_idempotent_on_the_description(
        self, client: httpx.AsyncClient
    ) -> None:
        first = await create_job(client)
        second = await create_job(client)
        assert first == second

        listing = (await client.get("/api/jobs")).json()
        assert len(listing) == 1

    async def test_a_stub_description_is_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/jobs", json={"description": "Backend dev wanted"})
        assert response.status_code == 422

    async def test_missing_job_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/jobs/nope")).status_code == 404

    async def test_rubric_appears_after_the_first_run(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        assert (await client.get(f"/api/jobs/{job_id}")).json()["requirements"] == []

        document_id = await upload_resume(client)
        await run_to_completion(client, job_id, [document_id])

        requirements = (await client.get(f"/api/jobs/{job_id}")).json()["requirements"]
        assert len(requirements) == 2
        assert sum(r["weight"] for r in requirements) == pytest.approx(100.0)


class TestDocuments:
    async def test_upload_and_fetch(self, client: httpx.AsyncClient) -> None:
        document_id = await upload_resume(client)
        response = await client.get(f"/api/documents/{document_id}")
        assert response.status_code == 200
        assert response.json()["filename"] == "priya.txt"

    async def test_upload_is_idempotent_on_content(self, client: httpx.AsyncClient) -> None:
        first = await client.post(
            "/api/documents", files={"files": ("a.txt", RESUME.encode(), "text/plain")}
        )
        second = await client.post(
            "/api/documents", files={"files": ("b.txt", RESUME.encode(), "text/plain")}
        )

        assert first.json()["uploaded"][0]["created"] is True
        assert second.json()["uploaded"][0]["created"] is False
        assert (
            first.json()["uploaded"][0]["document"]["id"]
            == second.json()["uploaded"][0]["document"]["id"]
        )

    async def test_a_bad_file_does_not_fail_the_batch(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/documents",
            files=[
                ("files", ("good.txt", RESUME.encode(), "text/plain")),
                ("files", ("bad.exe", b"binary junk", "application/octet-stream")),
                ("files", ("empty.txt", b"", "text/plain")),
            ],
        )
        body = response.json()
        assert len(body["uploaded"]) == 1
        assert len(body["rejected"]) == 2
        assert {item["filename"] for item in body["rejected"]} == {"bad.exe", "empty.txt"}

    async def test_text_endpoint_returns_the_offset_map(self, client: httpx.AsyncClient) -> None:
        document_id = await upload_resume(client)
        body = (await client.get(f"/api/documents/{document_id}/text")).json()

        assert "Kubernetes" in body["text"]
        assert body["blocks"]
        block = body["blocks"][0]
        assert body["text"][block["span"]["start"] : block["span"]["end"]].strip()

    async def test_raw_file_is_retained(self, client: httpx.AsyncClient) -> None:
        document_id = await upload_resume(client)
        response = await client.get(f"/api/documents/{document_id}/raw")
        assert response.status_code == 200
        assert b"PRIYA RAMAN" in response.content

    async def test_missing_document_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/documents/nope")).status_code == 404


class TestRuns:
    async def test_run_returns_202_immediately(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        document_id = await upload_resume(client)

        response = await client.post(
            "/api/runs", json={"job_id": job_id, "document_ids": [document_id]}
        )
        assert response.status_code == 202
        assert response.json()["status"] in ("pending", "running")

    async def test_run_completes_and_produces_a_shortlist(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        document_id = await upload_resume(client)
        run_id = await run_to_completion(client, job_id, [document_id])

        body = (await client.get(f"/api/runs/{run_id}/shortlist")).json()
        assert body["run"]["status"] == "completed"
        assert len(body["entries"]) == 1

        entry = body["entries"][0]
        assert entry["score"] > 0
        assert entry["score_low"] <= entry["score"] <= entry["score_high"]

    async def test_blind_mode_anonymises_the_candidate_label(
        self, client: httpx.AsyncClient
    ) -> None:
        job_id = await create_job(client)
        document_id = await upload_resume(client)
        run_id = await run_to_completion(client, job_id, [document_id], blind_mode=True)

        entry = (await client.get(f"/api/runs/{run_id}/shortlist")).json()["entries"][0]
        assert entry["candidate_label"].startswith("candidate-")
        assert "priya" not in entry["candidate_label"].lower()

    async def test_unknown_job_is_404(self, client: httpx.AsyncClient) -> None:
        document_id = await upload_resume(client)
        response = await client.post(
            "/api/runs", json={"job_id": "missing", "document_ids": [document_id]}
        )
        assert response.status_code == 404

    async def test_unknown_document_is_404(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        response = await client.post(
            "/api/runs", json={"job_id": job_id, "document_ids": ["missing"]}
        )
        assert response.status_code == 404

    async def test_empty_document_list_is_rejected(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        response = await client.post("/api/runs", json={"job_id": job_id, "document_ids": []})
        assert response.status_code == 422

    async def test_runs_are_listed_against_their_job(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        document_id = await upload_resume(client)
        await run_to_completion(client, job_id, [document_id])

        runs = (await client.get(f"/api/jobs/{job_id}/runs")).json()
        assert len(runs) == 1


class TestAssessmentDetail:
    @pytest_asyncio.fixture
    async def detail(self, client: httpx.AsyncClient) -> dict:
        job_id = await create_job(client)
        document_id = await upload_resume(client)
        run_id = await run_to_completion(client, job_id, [document_id])

        assessment_id = (await client.get(f"/api/runs/{run_id}/shortlist")).json()["entries"][0][
            "id"
        ]
        return (await client.get(f"/api/assessments/{assessment_id}")).json()

    async def test_includes_every_requirement(self, detail: dict) -> None:
        assert len(detail["requirements"]) == 2
        for requirement in detail["requirements"]:
            assert requirement["verdict"]
            assert requirement["max_points"] > 0

    async def test_citations_are_verified_at_read_time(self, detail: dict) -> None:
        cited = [
            citation
            for requirement in detail["requirements"]
            for citation in requirement["citations"]
        ]
        assert cited, "expected at least one citation"
        assert all(citation["verified"] for citation in cited)

    async def test_citation_quotes_come_from_the_stored_document(self, detail: dict) -> None:
        cited = [
            citation
            for requirement in detail["requirements"]
            for citation in requirement["citations"]
        ]
        assert any(
            "Kubernetes" in citation["quote"] or "p99" in citation["quote"] for citation in cited
        )

    async def test_interview_questions_are_returned(self, detail: dict) -> None:
        assert detail["questions"]
        assert detail["questions"][0]["rationale"]

    async def test_missing_assessment_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/assessments/nope")).status_code == 404


class TestProgressChannel:
    async def test_a_late_subscriber_gets_the_current_state(self) -> None:
        channel = ProgressChannel("run-1")
        channel.publish(
            RunProgress(
                run_id="run-1", status="completed", stage="done", total=1, completed=1, failed=0
            )
        )

        events = [event async for event in channel.subscribe()]
        assert len(events) == 1
        assert events[0].is_terminal

    async def test_subscribers_receive_published_events(self) -> None:
        channel = ProgressChannel("run-1")
        received: list[RunProgress] = []

        async def consume() -> None:
            async for event in channel.subscribe():
                received.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)

        channel.publish(
            RunProgress(
                run_id="run-1", status="running", stage="screening", total=2, completed=1, failed=0
            )
        )
        channel.publish(
            RunProgress(
                run_id="run-1", status="completed", stage="done", total=2, completed=2, failed=0
            )
        )
        await asyncio.wait_for(task, timeout=2.0)

        assert len(received) == 2
        assert received[-1].status == "completed"

    async def test_a_slow_subscriber_never_blocks_the_pipeline(self) -> None:
        channel = ProgressChannel("run-1")
        stalled: list[RunProgress] = []

        async def never_reads() -> None:
            async for event in channel.subscribe():
                stalled.append(event)
                await asyncio.sleep(3600)

        task = asyncio.create_task(never_reads())
        await asyncio.sleep(0)

        for index in range(200):
            channel.publish(
                RunProgress(
                    run_id="run-1",
                    status="running",
                    stage=f"step {index}",
                    total=200,
                    completed=index,
                    failed=0,
                )
            )
        assert channel.latest is not None
        assert channel.latest.completed == 199

        task.cancel()


class TestEventStream:
    async def test_stream_reports_a_finished_run(self, client: httpx.AsyncClient) -> None:
        job_id = await create_job(client)
        document_id = await upload_resume(client)
        run_id = await run_to_completion(client, job_id, [document_id])

        async with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            frames = []
            async for line in response.aiter_lines():
                frames.append(line)
                if line.startswith("event: done"):
                    break

        payloads = [line for line in frames if line.startswith("data: ")]
        assert payloads
        first = json.loads(payloads[0].removeprefix("data: "))
        assert first["run_id"] == run_id
        assert first["status"] == "completed"

    async def test_stream_for_a_missing_run_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/runs/nope/events")).status_code == 404


class TestFrontendMount:
    def test_no_build_means_no_mount(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HIRELENS_STATIC_DIR", str(tmp_path / "absent"))
        assert find_static_dir() is None

    def test_a_directory_without_index_html_is_not_a_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "assets").mkdir()
        monkeypatch.setenv("HIRELENS_STATIC_DIR", str(tmp_path))
        assert find_static_dir() is None

    def test_a_build_is_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
        monkeypatch.setenv("HIRELENS_STATIC_DIR", str(tmp_path))
        assert find_static_dir() == tmp_path

    async def test_api_routes_survive_the_root_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
        monkeypatch.setenv("HIRELENS_STATIC_DIR", str(tmp_path))

        app = create_app(database_url="sqlite+aiosqlite:///:memory:")
        async with (
            httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as api_client,
            app.router.lifespan_context(app),
        ):
            health = await api_client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] in ("ok", "degraded")

            index = await api_client.get("/")
            assert index.status_code == 200
            assert "dashboard" in index.text
