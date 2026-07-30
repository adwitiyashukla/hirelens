"""Phase 5 tests: metrics, baselines, golden set, labels, and the regression gate.

The metrics tests are checked against values computed by hand or against known
properties, because a metric implementation that is silently wrong would make
every number the project reports wrong in the same direction, and nothing else
would catch it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from hirelens.evals.baselines import BM25Baseline, KeywordOverlapBaseline, RandomBaseline
from hirelens.evals.golden import build_golden_set
from hirelens.evals.labels import (
    TIER_ORDER,
    Label,
    LabelSet,
    Tier,
    check_label_quality,
)
from hirelens.evals.metrics import (
    Distribution,
    bootstrap,
    inversion_rate,
    kendall_tau_b,
    mean_absolute_error,
    pearson,
    percentile,
    rank_with_ties,
    spearman,
    spearman_ci,
    top_k_precision,
)
from hirelens.evals.profiles import Demographics
from hirelens.evals.report import check_regression, to_console, to_markdown
from hirelens.evals.runner import EvalReport, MetricBlock, _PairResult, _pool_by_job

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestRanking:
    def test_ranks_are_one_based(self) -> None:
        assert rank_with_ties([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]

    def test_ties_share_the_average_rank(self) -> None:
        """Otherwise the coefficient depends on the arbitrary order of equal items."""
        assert rank_with_ties([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]

    def test_a_block_of_ties_averages_correctly(self) -> None:
        assert rank_with_ties([1.0, 1.0, 1.0, 1.0]) == [2.5, 2.5, 2.5, 2.5]


class TestCorrelations:
    def test_perfect_agreement(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_monotonic_but_nonlinear_still_correlates_perfectly(self) -> None:
        """Spearman is rank-based, so any monotonic transform must give 1.0."""
        xs = [1.0, 2.0, 3.0, 4.0]
        assert spearman(xs, [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)

    def test_no_variance_returns_zero_rather_than_nan(self) -> None:
        assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_tie_corrected_against_a_hand_computed_value(self) -> None:
        """The naive 1 - 6*sum(d^2)/(n(n^2-1)) formula is wrong under ties."""
        system = [10.0, 20.0, 30.0, 40.0]
        human = [1.0, 1.0, 2.0, 3.0]
        # Ranks: system [1,2,3,4], human [1.5,1.5,3,4]. Pearson on those.
        assert spearman(system, human) == pytest.approx(pearson([1, 2, 3, 4], [1.5, 1.5, 3, 4]))

    def test_kendall_perfect_agreement(self) -> None:
        assert kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_kendall_handles_ties(self) -> None:
        value = kendall_tau_b([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 2.0])
        assert 0.0 < value < 1.0

    def test_kendall_is_less_moved_by_one_outlier_than_spearman(self) -> None:
        human = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        # Move the best candidate to last place.
        system = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]
        assert kendall_tau_b(system, human) > spearman(system, human)


class TestInversionRate:
    def test_perfect_order_has_no_inversions(self) -> None:
        assert inversion_rate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_reversed_order_inverts_everything(self) -> None:
        assert inversion_rate([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == 1.0

    def test_pairs_the_human_tied_are_excluded(self) -> None:
        """There is no wrong answer for a pair the human considered equal."""
        assert inversion_rate([1.0, 2.0], [5.0, 5.0]) == 0.0


class TestTopKPrecision:
    def test_top_three_all_correct(self) -> None:
        system = [9.0, 8.0, 7.0, 1.0, 0.0]
        human = [4.0, 4.0, 3.0, 1.0, 0.0]
        assert top_k_precision(system, human, k=3) == pytest.approx(1.0)

    def test_top_three_all_wrong(self) -> None:
        """Exactly reversed: our top three are the human's bottom three."""
        human = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        system = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert top_k_precision(system, human, k=3) == pytest.approx(0.0)

    def test_partial_overlap_scores_between(self) -> None:
        system = [0.0, 1.0, 2.0, 8.0, 9.0]
        human = [4.0, 4.0, 4.0, 0.0, 0.0]
        # Our top three are indices 4, 3, 2; only index 2 is in the human top tier.
        assert top_k_precision(system, human, k=3) == pytest.approx(1 / 3)

    def test_k_larger_than_the_set_is_safe(self) -> None:
        assert top_k_precision([1.0, 2.0], [1.0, 2.0], k=10) == pytest.approx(1.0)


class TestBootstrap:
    def test_interval_contains_the_point_estimate(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ys = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0]
        estimate = spearman_ci(xs, ys)
        assert estimate.low <= estimate.value <= estimate.high

    def test_is_reproducible(self) -> None:
        """A metric that moves when nothing changed cannot gate anything."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ys = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]
        assert spearman_ci(xs, ys) == spearman_ci(xs, ys)

    def test_tiny_samples_report_the_full_range(self) -> None:
        """Rather than a falsely narrow interval that invites overclaiming."""
        estimate = spearman_ci([1.0, 2.0], [1.0, 2.0])
        assert (estimate.low, estimate.high) == (-1.0, 1.0)

    def test_noisier_data_gives_a_wider_interval(self) -> None:
        clean = spearman_ci([1.0, 2, 3, 4, 5, 6, 7, 8], [1.0, 2, 3, 4, 5, 6, 7, 8])
        noisy = spearman_ci([1.0, 2, 3, 4, 5, 6, 7, 8], [5.0, 2, 8, 1, 7, 3, 6, 4])
        assert noisy.width > clean.width

    def test_works_with_any_statistic(self) -> None:
        estimate = bootstrap([1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5], kendall_tau_b)
        assert estimate.value == pytest.approx(1.0)


class TestDistribution:
    def test_percentiles(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 0.5) == 3.0
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 5.0

    def test_summary(self) -> None:
        d = Distribution.of([1.0, 2.0, 3.0, 4.0])
        assert d.mean == 2.5
        assert d.n == 4

    def test_empty_is_safe(self) -> None:
        assert Distribution.of([]).n == 0

    def test_mae(self) -> None:
        assert mean_absolute_error([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


class TestGoldenSet:
    def test_has_profiles_and_jobs(self) -> None:
        golden = build_golden_set()
        assert len(golden.profiles) >= 10
        assert len(golden.jobs) == 3
        assert golden.pair_count >= 30

    def test_candidate_ids_are_unique(self) -> None:
        ids = [p.candidate_id for p in build_golden_set().profiles]
        assert len(ids) == len(set(ids))

    def test_quality_tiers_span_a_range(self) -> None:
        """A set where everyone is mediocre cannot distinguish good ranking from bad."""
        tiers = {p.quality for p in build_golden_set().profiles}
        assert len(tiers) >= 3

    def test_every_profile_documents_why_it_exists(self) -> None:
        for profile in build_golden_set().profiles:
            assert profile.notes.strip(), profile.candidate_id

    def test_rendering_is_deterministic(self) -> None:
        """Content-addressed ids and a warm cache depend on this."""
        golden = build_golden_set()
        for profile in golden.profiles:
            assert profile.render() == profile.render()

    def test_rendered_resumes_look_like_resumes(self) -> None:
        for profile in build_golden_set().profiles:
            text = profile.render()
            assert "EDUCATION" in text
            assert profile.demographics.name.upper() in text

    def test_round_trips_through_json(self) -> None:
        golden = build_golden_set()
        restored = type(golden).model_validate(json.loads(golden.model_dump_json()))
        assert restored.profiles[0].render() == golden.profiles[0].render()


class TestDemographicPerturbation:
    def test_swapping_demographics_changes_only_demographics(self) -> None:
        """The entire Phase 6 fairness claim rests on this being true."""
        profile = build_golden_set().profiles[0]
        swapped = profile.with_demographics(
            Demographics(name="Different Person", university="Other College", location="Elsewhere")
        )

        assert swapped.roles == profile.roles
        assert swapped.projects == profile.projects
        assert swapped.skills == profile.skills
        assert swapped.headline == profile.headline

    def test_swapped_render_differs_only_in_the_identity_lines(self) -> None:
        profile = build_golden_set().profiles[0]
        swapped = profile.with_demographics(
            Demographics(name="Different Person", university="Other College", location="Elsewhere")
        )

        original_lines = profile.render().splitlines()
        swapped_lines = swapped.render().splitlines()
        assert len(original_lines) == len(swapped_lines)

        differing = [(a, b) for a, b in zip(original_lines, swapped_lines, strict=True) if a != b]
        # Name line, contact line, education line. Nothing about ability.
        assert len(differing) == 3
        for original, _ in differing:
            assert "Kafka" not in original
            assert "Kubernetes" not in original


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    RESUMES: ClassVar[list[str]] = [
        "Deployed services on Kubernetes and ran Kafka consumers in production for three years.",
        "Designed marketing brochures and managed social media campaigns.",
        "Kubernetes Kafka Terraform Go gRPC microservices distributed systems",
    ]
    JOB = "We need someone with Kubernetes and Kafka experience running services in production."

    def test_keyword_baseline_ranks_the_stuffed_resume_highly(self) -> None:
        """This is the failure mode the baseline exists to demonstrate."""
        scores = KeywordOverlapBaseline().score(self.JOB, self.RESUMES)
        assert scores[2] >= scores[1]

    def test_bm25_baseline_prefers_relevant_over_irrelevant(self) -> None:
        scores = BM25Baseline().score(self.JOB, self.RESUMES)
        assert scores[0] > scores[1]

    def test_bm25_handles_an_empty_corpus(self) -> None:
        assert BM25Baseline().score(self.JOB, []) == []

    def test_random_baseline_is_reproducible(self) -> None:
        assert RandomBaseline().score(self.JOB, self.RESUMES) == RandomBaseline().score(
            self.JOB, self.RESUMES
        )

    def test_chance_ceiling_is_well_above_zero_on_small_sets(self) -> None:
        """The reason a bare correlation on 12 candidates proves nothing."""
        human = [float(i) for i in range(12)]
        _, ceiling = RandomBaseline().expected_correlation(human, spearman)
        assert ceiling > 0.3

    def test_chance_ceiling_shrinks_as_the_set_grows(self) -> None:
        small = RandomBaseline().expected_correlation([float(i) for i in range(8)], spearman)[1]
        large = RandomBaseline().expected_correlation([float(i) for i in range(60)], spearman)[1]
        assert large < small


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabelSet:
    def test_upsert_replaces_rather_than_duplicating(self) -> None:
        labels = LabelSet()
        labels.upsert(Label.create("backend", "c01", Tier.YES))
        labels.upsert(Label.create("backend", "c01", Tier.STRONG_YES))

        assert len(labels.labels) == 1
        assert labels.get("backend", "c01").tier is Tier.STRONG_YES

    def test_missing_lists_unlabelled_pairs(self) -> None:
        labels = LabelSet()
        labels.upsert(Label.create("backend", "c01", Tier.YES))
        missing = labels.missing(["backend", "ml"], ["c01", "c02"])
        assert ("backend", "c01") not in missing
        assert ("ml", "c02") in missing

    def test_tier_values_are_ordered(self) -> None:
        from hirelens.evals.labels import TIER_VALUES

        values = [TIER_VALUES[tier] for tier in TIER_ORDER]
        assert values == sorted(values)

    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        labels = LabelSet()
        labels.upsert(Label.create("backend", "c01", Tier.YES, rationale="ships to prod"))
        path = tmp_path / "labels.json"
        labels.save(path)

        restored = LabelSet.load(path)
        assert restored.get("backend", "c01").rationale == "ships to prod"

    def test_loading_a_missing_file_gives_an_empty_set(self, tmp_path: Path) -> None:
        assert LabelSet.load(tmp_path / "nope.json").labels == []


class TestLabelQuality:
    def test_flags_a_set_with_no_variance(self) -> None:
        """Every candidate in one tier means there is no ranking to reproduce."""
        labels = LabelSet()
        for index in range(6):
            labels.upsert(Label.create("backend", f"c{index:02d}", Tier.YES, rationale="ok"))

        codes = {w.code for w in check_label_quality(labels, ["backend"])}
        assert "no_variance" in codes

    def test_flags_too_few_labels(self) -> None:
        labels = LabelSet()
        labels.upsert(Label.create("backend", "c01", Tier.YES))
        codes = {w.code for w in check_label_quality(labels, ["backend"])}
        assert "too_few_labels" in codes

    def test_flags_missing_rationales(self) -> None:
        labels = LabelSet()
        for index, tier in enumerate(TIER_ORDER * 2):
            labels.upsert(Label.create("backend", f"c{index:02d}", tier))
        codes = {w.code for w in check_label_quality(labels, ["backend"])}
        assert "missing_rationales" in codes

    def test_a_healthy_set_produces_no_warnings(self) -> None:
        labels = LabelSet()
        for index, tier in enumerate(TIER_ORDER * 2):
            labels.upsert(
                Label.create("backend", f"c{index:02d}", tier, rationale="considered reason")
            )
        assert check_label_quality(labels, ["backend"]) == []


# ---------------------------------------------------------------------------
# Pooling and reporting
# ---------------------------------------------------------------------------


def pair(job: str, candidate: str, score: float, human: float) -> _PairResult:
    return _PairResult(
        job_id=job,
        candidate_id=candidate,
        score=score,
        human=human,
        grounding=1.0,
        citation_validity=1.0,
        agreement=1.0,
        band=0.0,
        ambiguous=0,
        requirements=5,
        unmet_must_haves=0,
        elapsed_s=1.0,
    )


class TestPooling:
    def test_scores_are_standardised_within_each_job(self) -> None:
        """An easy job's inflated scores must not dominate the pooled coefficient."""
        results = [
            pair("easy", "c01", 90.0, 4.0),
            pair("easy", "c02", 85.0, 3.0),
            pair("hard", "c01", 30.0, 4.0),
            pair("hard", "c02", 20.0, 3.0),
        ]
        system, human = _pool_by_job(results)
        # Both jobs rank identically, so pooled agreement must be perfect despite
        # the raw scales being completely different.
        assert spearman(system, human) == pytest.approx(1.0)

    def test_pooling_preserves_pair_count(self) -> None:
        results = [pair("a", "c01", 1.0, 1.0), pair("a", "c02", 2.0, 2.0)]
        system, human = _pool_by_job(results)
        assert len(system) == len(human) == 2


class TestReportRendering:
    def build(self, rho: float = 0.8, baseline_rho: float = 0.4) -> EvalReport:
        return EvalReport(
            provider="ollama",
            model="qwen3:4b",
            embedder="hashing-384d",
            self_consistency_k=5,
            blind_mode=True,
            pairs_evaluated=36,
            pooled=MetricBlock(
                label="pooled",
                n=36,
                spearman=rho,
                spearman_low=rho - 0.2,
                spearman_high=rho + 0.1,
                kendall=rho - 0.1,
                kendall_low=rho - 0.3,
                kendall_high=rho,
                inversion_rate=0.15,
                top_3_precision=0.67,
            ),
            baselines={
                "bm25": MetricBlock(
                    label="bm25",
                    n=36,
                    spearman=baseline_rho,
                    spearman_low=baseline_rho - 0.2,
                    spearman_high=baseline_rho + 0.2,
                    kendall=baseline_rho,
                    kendall_low=0.0,
                    kendall_high=0.5,
                    inversion_rate=0.3,
                    top_3_precision=0.33,
                )
            },
            random_ceiling_95=0.35,
        )

    def test_markdown_includes_the_baseline_row(self) -> None:
        """Omitting it would make the headline number unanchored."""
        markdown = to_markdown(self.build())
        assert "bm25" in markdown
        assert "HireLens" in markdown

    def test_markdown_includes_confidence_intervals(self) -> None:
        assert "95% CI" in to_markdown(self.build())

    def test_console_marks_a_failure_to_beat_baselines(self) -> None:
        text = to_console(self.build(rho=0.2, baseline_rho=0.5))
        assert "beats every baseline : NO" in text

    def test_empty_report_explains_what_to_do(self) -> None:
        empty = EvalReport(
            provider="ollama",
            model="m",
            embedder="e",
            self_consistency_k=5,
            blind_mode=True,
            pairs_evaluated=0,
        )
        assert "label" in to_console(empty).lower()


class TestRegressionGate:
    def build(self, rho: float, *, baseline_rho: float = 0.3, validity: float = 0.98) -> EvalReport:
        from hirelens.evals.runner import QualityBlock

        return EvalReport(
            provider="ollama",
            model="m",
            embedder="e",
            self_consistency_k=5,
            blind_mode=True,
            pairs_evaluated=36,
            pooled=MetricBlock(
                label="pooled",
                n=36,
                spearman=rho,
                spearman_low=rho - 0.2,
                spearman_high=rho + 0.1,
                kendall=rho,
                kendall_low=0.0,
                kendall_high=1.0,
                inversion_rate=0.2,
                top_3_precision=0.67,
            ),
            baselines={
                "bm25": MetricBlock(
                    label="bm25",
                    n=36,
                    spearman=baseline_rho,
                    spearman_low=0.0,
                    spearman_high=0.6,
                    kendall=baseline_rho,
                    kendall_low=0.0,
                    kendall_high=0.5,
                    inversion_rate=0.3,
                    top_3_precision=0.33,
                )
            },
            random_ceiling_95=0.25,
            quality=QualityBlock(
                mean_grounding_rate=0.95,
                mean_citation_validity=validity,
                mean_sample_agreement=0.9,
                mean_confidence_band=4.0,
                ambiguous_requirement_rate=0.1,
                unmet_must_have_rate=0.2,
            ),
        )

    def test_a_healthy_run_passes(self) -> None:
        assert check_regression(self.build(0.8), None).passed

    def test_failing_to_beat_a_baseline_fails_the_gate(self) -> None:
        result = check_regression(self.build(0.3, baseline_rho=0.6), None)
        assert not result.passed
        assert any("baseline" in failure for failure in result.failures)

    def test_a_result_indistinguishable_from_chance_fails(self) -> None:
        result = check_regression(self.build(0.2, baseline_rho=0.1), None)
        assert not result.passed
        assert any("noise" in failure for failure in result.failures)

    def test_low_citation_validity_fails(self) -> None:
        """The grounding claim is the core of the project, so it is a hard floor."""
        result = check_regression(self.build(0.8, validity=0.5), None)
        assert not result.passed
        assert any("Citation validity" in failure for failure in result.failures)

    def test_a_large_drop_against_the_baseline_fails(self) -> None:
        result = check_regression(self.build(0.60), self.build(0.80))
        assert not result.passed
        assert any("dropped" in failure for failure in result.failures)

    def test_a_small_drop_is_tolerated(self) -> None:
        """A gate that fires on noise gets disabled within a week."""
        result = check_regression(self.build(0.78), self.build(0.80))
        assert result.passed

    def test_an_improvement_passes_and_is_noted(self) -> None:
        result = check_regression(self.build(0.90), self.build(0.80))
        assert result.passed
        assert any("up" in note for note in result.notes)

    def test_report_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        self.build(0.8).save(path)
        assert EvalReport.load(path).pooled.spearman == pytest.approx(0.8)
