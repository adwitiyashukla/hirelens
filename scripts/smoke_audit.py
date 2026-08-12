from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from hirelens.audit import Axis, FairnessAudit, check_audit, to_console
from hirelens.config import Provider, Settings
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve import HashingEmbedder

sys.path.insert(0, str(Path(__file__).parent))
from smoke_eval import (
    RUBRICS,
    _extract_education,
    _extract_work,
    _judge,
    _section_body,
)

FAVOURED = ("Stanford", "Indian Institute of Technology")


class StubProvider(LLMProvider):
    name = "stub"
    model = "stub"

    def __init__(self, *, bias_points: int = 0) -> None:
        self.bias_points = bias_points

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        convo = "\n".join(m.content for m in request.messages)
        prompt = request.messages[-1].content

        if "Compile the following job description" in convo:
            key = next(k for k in RUBRICS if k in convo)
            payload: dict = {
                "role_title": key,
                "seniority": "senior",
                "requirements": [
                    {"text": t, "kind": k, "category": "experience", "evidence_hint": h}
                    for t, k, h in RUBRICS[key]
                ],
            }
        elif "REQUIREMENT:" in convo:
            payload = _judge(convo)
            evidence = convo.split("EVIDENCE RETRIEVED")[1]
            if self.bias_points and any(name in evidence for name in FAVOURED):
                payload["verdict"] = "strong"
        elif "Extract paid professional experience" in convo:
            payload = _extract_work(_section_body(prompt))
        elif "Extract formal education" in convo:
            payload = _extract_education(_section_body(prompt))
        else:
            payload = {}

        return CompletionResponse(
            content=json.dumps(payload), model=self.model, usage=Usage(120, 40)
        )

    async def aclose(self) -> None:
        return None


async def run_case(label: str, *, bias_points: int) -> bool:
    settings = Settings(
        llm_provider=Provider.OLLAMA,
        cache_enabled=True,
        cache_dir=Path(tempfile.mkdtemp()),
        blind_mode=True,
        self_consistency_k=2,
        max_demographic_drift=2.0,
        requests_per_minute=0,
    )
    client = LLMClient(StubProvider(bias_points=bias_points), settings=settings)
    audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

    report = await audit.run(
        job_id="backend",
        profile_ids=["c01", "c02", "c11"],
        axes=(Axis.GENDER, Axis.UNIVERSITY),
        variants_per_axis=3,
        both_modes=True,
        k=2,
    )

    print(f"\n{'#' * 76}\n# {label}\n{'#' * 76}")
    print(to_console(report))
    print("\nGate:")
    print(check_audit(report).render())
    await client.aclose()
    return report.passes


async def main() -> int:
    clean = await run_case("CASE 1: unbiased stand-in", bias_points=0)
    await run_case("CASE 2: stand-in rigged to reward elite institutions", bias_points=20)

    print(
        "\nNOTE: deterministic stand-in, not a real model. This validates that the audit "
        "detects injected bias and stays quiet when there is none."
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
