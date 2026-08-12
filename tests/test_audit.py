from __future__ import annotations

import json
from pathlib import Path

import pytest

from hirelens.audit.perturbations import (
    DEFAULT_AXES,
    ETHNICITY_VARIANTS,
    GENDER_VARIANTS,
    NULL_VARIANTS,
    UNIVERSITY_VARIANTS,
    Axis,
    build_plan,
    estimate_calls,
    variants_for,
)
from hirelens.audit.report import check_audit, to_console, to_markdown
from hirelens.audit.runner import AuditReport, AxisResult, FairnessAudit, Observation, _axis_result
from hirelens.config import Provider, Settings
from hirelens.evals.golden import build_golden_set
from hirelens.evals.profiles import Demographics
from hirelens.llm.base import CompletionRequest, CompletionResponse, LLMProvider, Usage
from hirelens.llm.client import LLMClient
from hirelens.retrieve.embeddings import HashingEmbedder


class TestPerturbationSets:
    def test_null_control_changes_nothing(self) -> None:
        base = Demographics(name="Original Person", university="Some College")
        for variant in NULL_VARIANTS:
            assert variant.apply(base) == base

    def test_two_controls_exist(self) -> None:
        assert len(NULL_VARIANTS) >= 2

    def test_gender_variants_share_a_surname(self) -> None:
        surnames = {v.overrides["name"].split()[-1] for v in GENDER_VARIANTS}
        assert len(surnames) == 1

    def test_gender_variants_move_pronouns_with_the_name(self) -> None:
        for variant in GENDER_VARIANTS:
            assert "pronouns" in variant.overrides

    def test_ethnicity_variants_only_change_the_name(self) -> None:
        for variant in ETHNICITY_VARIANTS:
            assert set(variant.overrides) == {"name"}

    def test_university_variants_only_change_the_institution(self) -> None:
        for variant in UNIVERSITY_VARIANTS:
            assert set(variant.overrides) == {"university"}

    def test_every_variant_declares_a_group(self) -> None:
        for variant in build_plan(DEFAULT_AXES):
            assert variant.group

    def test_variants_never_touch_ability_fields(self) -> None:
        allowed = {"name", "pronouns", "university", "location", "email"}
        for variant in build_plan(DEFAULT_AXES):
            assert set(variant.overrides) <= allowed


class TestPlanBuilding:
    def test_control_is_always_first(self) -> None:
        plan = build_plan(DEFAULT_AXES, variants_per_axis=2)
        assert plan[0].axis is Axis.NULL

    def test_truncation_never_drops_the_control(self) -> None:
        plan = build_plan(DEFAULT_AXES, variants_per_axis=1)
        assert sum(1 for v in plan if v.is_control) == len(NULL_VARIANTS)

    def test_truncation_limits_each_axis(self) -> None:
        plan = build_plan((Axis.GENDER, Axis.ETHNICITY), variants_per_axis=2)
        assert sum(1 for v in plan if v.axis is Axis.GENDER) == 2
        assert sum(1 for v in plan if v.axis is Axis.ETHNICITY) == 2

    def test_variants_for_preserves_order(self) -> None:
        assert variants_for(Axis.GENDER, limit=2) == GENDER_VARIANTS[:2]

    def test_estimate_scales_with_every_dimension(self) -> None:
        base = estimate_calls(profiles=2, variants=10, modes=1, self_consistency_k=2)
        assert estimate_calls(profiles=4, variants=10, modes=1, self_consistency_k=2) == base * 2
        assert estimate_calls(profiles=2, variants=10, modes=2, self_consistency_k=2) == base * 2


class TestProfilePerturbationIntegrity:
    def test_applying_a_variant_changes_only_identity_lines(self) -> None:
        profile = build_golden_set().profiles[0]
        variant = GENDER_VARIANTS[0]
        perturbed = profile.with_demographics(variant.apply(profile.demographics))

        original = profile.render().splitlines()
        changed = perturbed.render().splitlines()
        assert len(original) == len(changed)

        differing = [i for i, (a, b) in enumerate(zip(original, changed, strict=True)) if a != b]
        for index in differing:
            line = original[index]
            assert not any(
                token in line for token in ("Kafka", "Kubernetes", "latency", "EXPERIENCE")
            )

    def test_evidence_sections_are_byte_identical_across_every_variant(self) -> None:
        profile = build_golden_set().profiles[0]
        baseline = profile.render()
        evidence_start = baseline.index("EXPERIENCE")
        evidence_end = baseline.index("EDUCATION")
        reference = baseline[evidence_start:evidence_end]

        for variant in build_plan(DEFAULT_AXES):
            perturbed = profile.with_demographics(variant.apply(profile.demographics)).render()
            start = perturbed.index("EXPERIENCE")
            end = perturbed.index("EDUCATION")
            assert perturbed[start:end] == reference, variant.label


def obs(
    candidate: str,
    axis: str,
    label: str,
    group: str,
    score: float,
    *,
    blind: bool = True,
    meets: bool = True,
) -> Observation:
    return Observation(
        candidate_id=candidate,
        job_id="backend",
        axis=axis,
        variant_label=label,
        group=group,
        blind=blind,
        score=score,
        meets_must_haves=meets,
    )


class TestAxisResult:
    def test_drift_is_measured_within_a_profile(self) -> None:
        observations = [
            obs("c01", "gender", "male", "male", 80.0),
            obs("c01", "gender", "female", "female", 70.0),
            obs("c02", "gender", "male", "male", 40.0),
            obs("c02", "gender", "female", "female", 38.0),
        ]
        result = _axis_result(observations, "gender", blind=True)
        assert result is not None
        assert result.max_drift == pytest.approx(10.0)
        assert result.mean_drift == pytest.approx(6.0)

    def test_group_gap_compares_group_means(self) -> None:
        observations = [
            obs("c01", "gender", "male", "male", 80.0),
            obs("c01", "gender", "female", "female", 70.0),
            obs("c02", "gender", "male", "male", 60.0),
            obs("c02", "gender", "female", "female", 50.0),
        ]
        result = _axis_result(observations, "gender", blind=True)
        assert result is not None
        assert result.group_gap == pytest.approx(10.0)
        assert result.favoured_group == "male"
        assert result.disfavoured_group == "female"

    def test_no_drift_when_scores_are_identical(self) -> None:
        observations = [
            obs("c01", "gender", "male", "male", 75.0),
            obs("c01", "gender", "female", "female", 75.0),
        ]
        result = _axis_result(observations, "gender", blind=True)
        assert result is not None
        assert result.max_drift == 0.0
        assert result.group_gap == 0.0

    def test_must_have_flips_are_counted(self) -> None:
        observations = [
            obs("c01", "gender", "male", "male", 75.0, meets=True),
            obs("c01", "gender", "female", "female", 74.0, meets=False),
        ]
        result = _axis_result(observations, "gender", blind=True)
        assert result is not None
        assert result.must_have_flips == 1

    def test_blind_and_sighted_are_kept_separate(self) -> None:
        observations = [
            obs("c01", "gender", "male", "male", 80.0, blind=True),
            obs("c01", "gender", "female", "female", 80.0, blind=True),
            obs("c01", "gender", "male", "male", 80.0, blind=False),
            obs("c01", "gender", "female", "female", 60.0, blind=False),
        ]
        assert _axis_result(observations, "gender", blind=True).max_drift == 0.0
        assert _axis_result(observations, "gender", blind=False).max_drift == pytest.approx(20.0)


def axis_result(
    axis: str,
    *,
    blind: bool,
    drift: float,
    gap: float = 0.0,
    flips: int = 0,
    must_have_flips: int = 0,
) -> AxisResult:
    return AxisResult(
        axis=axis,
        blind=blind,
        observations=4,
        max_drift=drift,
        mean_drift=drift / 2,
        group_means={"a": 50.0 + gap, "b": 50.0},
        group_gap=gap,
        rank_flips=flips,
        must_have_flips=must_have_flips,
    )


def report_with(*results: AxisResult, noise: float = 1.0, threshold: float = 2.0) -> AuditReport:
    return AuditReport(
        provider="ollama",
        model="m",
        self_consistency_k=2,
        profiles_tested=3,
        variants_tested=10,
        job_id="backend",
        noise_floor=noise * 0.5,
        noise_floor_max=noise,
        axes=list(results),
        threshold=threshold,
    )


class TestExcessDrift:
    def test_drift_below_the_noise_floor_counts_as_zero(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=0.8), noise=1.5)
        assert report.excess_drift(report.axes[0]) == 0.0

    def test_excess_is_measured_above_the_floor(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=5.0), noise=1.5)
        assert report.excess_drift(report.axes[0]) == pytest.approx(3.5)

    def test_blind_mode_benefit_is_the_difference(self) -> None:
        report = report_with(
            axis_result("gender", blind=True, drift=0.5),
            axis_result("gender", blind=False, drift=7.5),
        )
        assert report.blind_mode_benefit == pytest.approx(7.0)


class TestAuditGate:
    def test_a_clean_audit_passes(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=1.2), noise=1.0)
        assert check_audit(report).passed

    def test_excess_blind_drift_fails(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=6.0), noise=1.0)
        result = check_audit(report)
        assert not result.passed
        assert any("gender" in failure for failure in result.failures)

    def test_a_must_have_flip_fails_even_within_the_score_threshold(self) -> None:
        report = report_with(
            axis_result("gender", blind=True, drift=1.0, must_have_flips=1), noise=1.0
        )
        result = check_audit(report)
        assert not result.passed
        assert any("hiring decision" in failure for failure in result.failures)

    def test_sighted_drift_is_a_note_not_a_failure(self) -> None:
        report = report_with(
            axis_result("gender", blind=True, drift=0.5),
            axis_result("gender", blind=False, drift=12.0),
            noise=1.0,
        )
        result = check_audit(report)
        assert result.passed
        assert any("blind mode OFF" in note for note in result.notes)

    def test_noise_above_the_threshold_fails_the_audit(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=1.0), noise=5.0)
        result = check_audit(report)
        assert not result.passed
        assert any("noise" in failure for failure in result.failures)

    def test_rank_flips_alone_are_a_note(self) -> None:
        report = report_with(axis_result("gender", blind=True, drift=1.0, flips=2), noise=1.0)
        result = check_audit(report)
        assert result.passed
        assert any("rank flip" in note for note in result.notes)

    def test_an_empty_audit_fails(self) -> None:
        assert not check_audit(report_with()).passed


class TestReportRendering:
    def test_markdown_states_the_method_and_the_control(self) -> None:
        markdown = to_markdown(
            report_with(
                axis_result("gender", blind=True, drift=0.5),
                axis_result("gender", blind=False, drift=6.0, gap=4.0),
            )
        )
        assert "null control" in markdown.lower()
        assert "Limitations" in markdown
        assert "byte-identical" in markdown

    def test_markdown_reports_limitations_honestly(self) -> None:
        markdown = to_markdown(report_with(axis_result("gender", blind=True, drift=0.5)))
        assert "not proof of fairness" in markdown

    def test_console_shows_the_noise_floor(self) -> None:
        text = to_console(report_with(axis_result("gender", blind=True, drift=0.5), noise=1.3))
        assert "noise floor" in text
        assert "1.30" in text

    def test_console_marks_failure(self) -> None:
        text = to_console(report_with(axis_result("gender", blind=True, drift=9.0), noise=1.0))
        assert "AUDIT FAILED" in text

    def test_report_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "fairness.json"
        report_with(axis_result("gender", blind=True, drift=1.0)).save(path)
        assert AuditReport.load(path).axes[0].max_drift == pytest.approx(1.0)


class BiasedProvider(LLMProvider):
    name = "biased"
    model = "biased-stub"

    def __init__(self, *, favoured: str = "Stanford", bias_points: int = 0) -> None:
        self.favoured = favoured
        self.bias_points = bias_points
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        convo = "\n".join(m.content for m in request.messages)
        self.calls += 1

        if "Compile the following job description" in convo:
            payload: dict = {
                "role_title": "Backend Engineer",
                "seniority": "senior",
                "requirements": [
                    {
                        "text": "Has run containers in production",
                        "kind": "must_have",
                        "category": "experience",
                        "evidence_hint": "Kubernetes containers deployed production",
                    },
                    {
                        "text": "Has an appropriate educational background",
                        "kind": "nice_to_have",
                        "category": "education",
                        "evidence_hint": "degree university institute",
                    },
                ],
            }
        elif "REQUIREMENT:" in convo:
            evidence = convo.split("EVIDENCE RETRIEVED")[1]
            has_signal = "Kubernetes" in evidence or "Kafka" in evidence
            verdict = "clear" if has_signal else "partial"
            if self.bias_points and self.favoured in evidence:
                verdict = "strong"
            ids = [line.split("]")[0][1:] for line in convo.split("\n") if line.startswith("[")][:1]
            payload = {"verdict": verdict, "reasoning": "stub", "evidence_unit_ids": ids}
        elif "Extract paid professional experience" in convo:
            payload = {
                "work": [
                    {
                        "company": {"value": "Northwind Payments", "quote": "Northwind Payments"},
                        "position": {
                            "value": "Senior Backend Engineer",
                            "quote": "Senior Backend Engineer",
                        },
                        "highlights": [
                            {
                                "value": "Ran services on Kubernetes",
                                "quote": "Ran the service on Kubernetes across three regions and carried the primary pager for two years.",
                            }
                        ],
                    }
                ]
            }
        elif "Extract formal education" in convo:
            body = convo.split("--- BEGIN RESUME TEXT ---")[-1].split("--- END")[0]
            institution = body.split(",")[-1].split("(")[0].strip() if "," in body else "Unknown"
            payload = {"education": [{"institution": {"value": institution, "quote": institution}}]}
        else:
            payload = {}

        return CompletionResponse(
            content=json.dumps(payload), model=self.model, usage=Usage(100, 30)
        )

    async def aclose(self) -> None:
        return None


def audit_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider=Provider.OLLAMA,
        cache_enabled=True,
        cache_dir=tmp_path,
        blind_mode=True,
        self_consistency_k=1,
        max_demographic_drift=2.0,
        requests_per_minute=0,
    )


class TestAuditEndToEnd:
    async def test_audit_runs_and_produces_a_report(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path)
        client = LLMClient(BiasedProvider(bias_points=0), settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        report = await audit.run(
            job_id="backend",
            profile_ids=["c01"],
            axes=(Axis.UNIVERSITY,),
            variants_per_axis=3,
            both_modes=False,
            k=1,
        )
        assert report.axes
        assert report.for_axis("university", blind=True) is not None
        await client.aclose()

    async def test_an_unbiased_model_produces_no_drift(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path)
        client = LLMClient(BiasedProvider(bias_points=0), settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        report = await audit.run(
            job_id="backend",
            profile_ids=["c01"],
            axes=(Axis.UNIVERSITY,),
            variants_per_axis=3,
            both_modes=False,
            k=1,
        )
        result = report.for_axis("university", blind=True)
        assert result is not None
        assert result.max_drift == 0.0
        assert check_audit(report).passed
        await client.aclose()

    async def test_audit_detects_a_deliberately_biased_model(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path).model_copy(update={"blind_mode": False})
        client = LLMClient(BiasedProvider(favoured="Stanford", bias_points=20), settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        report = await audit.run(
            job_id="backend",
            profile_ids=["c01"],
            axes=(Axis.UNIVERSITY,),
            variants_per_axis=3,
            both_modes=False,
            k=1,
        )

        result = report.for_axis("university", blind=False)
        assert result is not None
        assert result.max_drift > 0.0, "the audit failed to detect a model rigged to be biased"
        assert result.favoured_group == "elite"
        await client.aclose()

    async def test_blind_mode_suppresses_the_bias_the_model_has(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path)
        client = LLMClient(BiasedProvider(favoured="Stanford", bias_points=20), settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        report = await audit.run(
            job_id="backend",
            profile_ids=["c01"],
            axes=(Axis.UNIVERSITY,),
            variants_per_axis=3,
            both_modes=True,
            k=1,
        )

        blind = report.for_axis("university", blind=True)
        sighted = report.for_axis("university", blind=False)
        assert blind is not None and sighted is not None
        assert blind.max_drift < sighted.max_drift
        assert report.blind_mode_benefit > 0
        await client.aclose()

    async def test_the_audit_disables_caching_for_itself(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path)
        assert settings.cache_enabled is True

        provider = BiasedProvider(bias_points=0)
        client = LLMClient(provider, settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        await audit.run(
            job_id="backend",
            profile_ids=["c01"],
            axes=(Axis.NULL,),
            variants_per_axis=None,
            both_modes=False,
            k=1,
        )
        assert client.cache.hits == 0
        await client.aclose()

    async def test_unknown_job_is_rejected_clearly(self, tmp_path: Path) -> None:
        settings = audit_settings(tmp_path)
        client = LLMClient(BiasedProvider(), settings=settings)
        audit = FairnessAudit(settings=settings, embedder=HashingEmbedder(), client=client)

        with pytest.raises(ValueError, match="Unknown job"):
            await audit.run(job_id="nonexistent")
        await client.aclose()
