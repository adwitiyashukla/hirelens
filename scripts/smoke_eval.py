from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

from hirelens.config import Provider, Settings
from hirelens.evals import EvalHarness, Label, LabelSet, Tier, build_golden_set, to_console
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve import HashingEmbedder

GOLDEN = build_golden_set()

RUBRICS = {
    "payments platform": [
        (
            "Has run containerised workloads in production",
            "must_have",
            "Kubernetes containers deployed production cluster regions",
        ),
        (
            "Has worked with high-throughput event streaming",
            "must_have",
            "Kafka event streaming throughput events per day consumer",
        ),
        (
            "Has measurably improved system performance",
            "must_have",
            "latency reduced p99 milliseconds performance optimisation faster",
        ),
        (
            "Has owned services in production including on-call",
            "must_have",
            "on-call pager owned operated production incident reliability",
        ),
        (
            "Strong server-side programming",
            "must_have",
            "Go Java Rust backend services server-side API",
        ),
        (
            "Has built distributed systems",
            "nice_to_have",
            "distributed consensus Raft replication key-value store",
        ),
        (
            "Has contributed to open source",
            "nice_to_have",
            "open source stars merged upstream github contribution",
        ),
        ("Infrastructure as code", "nice_to_have", "Terraform infrastructure as code provisioning"),
    ],
    "Applied NLP": [
        (
            "Has built systems on large language models",
            "must_have",
            "language model GPT transformer retrieval augmented generation LLM",
        ),
        (
            "Has experience with retrieval or vector search",
            "must_have",
            "retrieval vector search embeddings BM25 dense FAISS reranker",
        ),
        (
            "Can design and run rigorous offline evaluation",
            "must_have",
            "evaluation harness nDCG labelled benchmark metric regression significance",
        ),
        ("Strong Python", "must_have", "Python PyTorch pandas scikit-learn transformers"),
        (
            "Has deployed an ML system to production users",
            "must_have",
            "shipped production daily users deployed model serving",
        ),
        (
            "Fine-tuning or model training",
            "nice_to_have",
            "fine-tuned trained classifier F1 baseline model",
        ),
        (
            "Responsible AI practice",
            "nice_to_have",
            "bias evaluation fairness responsible demographic audit",
        ),
        (
            "Published writing or open source in ML",
            "nice_to_have",
            "published paper award open source stars github",
        ),
    ],
    "Design Systems": [
        (
            "Strong React and TypeScript",
            "must_have",
            "React TypeScript components frontend interface",
        ),
        (
            "Has built or maintained a component library",
            "must_have",
            "design system component library Storybook reusable components teams",
        ),
        (
            "Working knowledge of accessibility",
            "must_have",
            "accessibility WCAG screen reader axe keyboard compliance",
        ),
        (
            "Collaborates directly with designers",
            "must_have",
            "designers Figma design tokens visual drift",
        ),
        (
            "Attention to cross-browser behaviour",
            "must_have",
            "cross-browser CSS visual detail styling",
        ),
        ("Design tokens or theming", "nice_to_have", "design tokens theming Figma shared"),
        (
            "Client-side performance work",
            "nice_to_have",
            "bundle size performance contentful paint lazy loading conversion",
        ),
        (
            "Testing of user interfaces",
            "nice_to_have",
            "Playwright Cypress end-to-end tests automated",
        ),
    ],
}

_ROLE_LINE = re.compile(
    r"^(?P<title>[^,]+), (?P<company>.+?) \((?P<start>[^ ]+) - (?P<end>[^)]+)\)$"
)


def _section_body(prompt: str) -> list[str]:
    body = prompt.split("--- BEGIN RESUME TEXT ---")[-1].split("--- END RESUME TEXT ---")[0]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _field(value: str, quote: str | None = None) -> dict:
    return {"value": value, "quote": quote if quote is not None else value}


def _extract_work(lines: list[str]) -> dict:
    roles: list[dict] = []
    for line in lines:
        match = _ROLE_LINE.match(line)
        if match:
            roles.append(
                {
                    "company": _field(match.group("company")),
                    "position": _field(match.group("title")),
                    "start_date": _field(match.group("start")),
                    "is_current": match.group("end") == "present",
                    "highlights": [],
                }
            )
        elif roles:
            roles[-1]["highlights"].append(_field(line))
    return {"work": roles}


def _extract_projects(lines: list[str]) -> dict:
    projects: list[dict] = []
    for line in lines:
        if ":" in line:
            name, rest = line.split(":", 1)
            url = ""
            description = rest.strip()
            if " - github.com/" in description:
                description, url = description.rsplit(" - ", 1)
            entry = {
                "name": _field(name.strip()),
                "description": _field(description.strip()),
                "technologies": [],
                "highlights": [],
            }
            if url:
                entry["url"] = _field(url.strip())
            projects.append(entry)
        elif projects:
            projects[-1]["highlights"].append(_field(line))
    return {"projects": projects}


def _extract_skills(lines: list[str]) -> dict:
    skills = [s.strip() for line in lines for s in line.split(",") if s.strip()]
    return {"skills": [{"name": _field(s), "category": ""} for s in skills]}


def _extract_education(lines: list[str]) -> dict:
    out = []
    for line in lines:
        if "," in line:
            degree, rest = line.split(",", 1)
            institution = rest.split("(")[0].strip()
            out.append({"institution": _field(institution), "degree": _field(degree.strip())})
    return {"education": out}


def _judge(convo: str) -> dict:
    evidence = convo.split("EVIDENCE RETRIEVED")[1].lower()
    requirement = convo.split("REQUIREMENT:")[1].split("\n")[0].lower()
    terms = [w.strip(",.") for w in requirement.split() if len(w) > 4]
    hits = sum(1 for term in terms if term.rstrip("s") in evidence)
    ratio = hits / max(len(terms), 1)

    verdict = (
        "strong"
        if ratio >= 0.55
        else "clear"
        if ratio >= 0.38
        else "partial"
        if ratio >= 0.22
        else "weak"
        if ratio > 0.08
        else "none"
    )
    ids = [line.split("]")[0][1:] for line in convo.split("\n") if line.startswith("[")][:2]
    return {
        "verdict": verdict,
        "reasoning": f"Term overlap {ratio:.0%} between requirement and evidence.",
        "evidence_unit_ids": ids if verdict != "none" else [],
    }


class HeuristicProvider(LLMProvider):
    name = "heuristic"
    model = "heuristic-stub"

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
        elif "Extract paid professional experience" in convo:
            payload = _extract_work(_section_body(prompt))
        elif "Extract personal, academic" in convo:
            payload = _extract_projects(_section_body(prompt))
        elif "Extract individual skills" in convo:
            payload = _extract_skills(_section_body(prompt))
        elif "Extract formal education" in convo:
            payload = _extract_education(_section_body(prompt))
        else:
            payload = {}

        return CompletionResponse(
            content=json.dumps(payload), model=self.model, usage=Usage(150, 45)
        )

    async def aclose(self) -> None:
        return None


def placeholder_labels() -> LabelSet:
    tiers = {"strong": Tier.STRONG_YES, "solid": Tier.YES, "mixed": Tier.MAYBE, "weak": Tier.NO}
    labels = LabelSet()
    for job in GOLDEN.jobs:
        for profile in GOLDEN.profiles:
            if profile.target_role == job.job_id:
                tier = tiers.get(str(profile.quality), Tier.MAYBE)
            elif not profile.target_role:
                tier = Tier.MAYBE if str(profile.quality) == "mixed" else Tier.NO
            else:
                tier = Tier.STRONG_NO
            labels.upsert(
                Label.create(job.job_id, profile.candidate_id, tier, rationale="placeholder")
            )
    return labels


async def main() -> int:
    settings = Settings(
        llm_provider=Provider.OLLAMA,
        cache_enabled=True,
        cache_dir=Path(tempfile.mkdtemp()),
        blind_mode=True,
        self_consistency_k=3,
        requests_per_minute=0,
    )
    client = LLMClient(HeuristicProvider(), settings=settings)
    harness = EvalHarness(settings=settings, embedder=HashingEmbedder(), client=client)

    report = await harness.run(placeholder_labels(), top_k=4)
    print(to_console(report))
    print(
        "\nNOTE: heuristic stand-in and placeholder labels. This validates the "
        "harness plumbing, not HireLens quality."
    )
    await client.aclose()
    return 0 if report.pairs_evaluated else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
