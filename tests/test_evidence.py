from __future__ import annotations

import pytest
from pydantic import ValidationError

from hirelens.schemas.evidence import Citation, Cited, EvidenceUnit, Span, VerificationResult

DOC = "Built a distributed task queue in Go handling 40k jobs per minute at Acme Corp."
DOC_ID = "doc-1"


def span_of(fragment: str) -> Span:
    start = DOC.index(fragment)
    return Span(start=start, end=start + len(fragment))


class TestSpan:
    def test_length_and_slice(self) -> None:
        span = Span(start=8, end=19)
        assert len(span) == 11
        assert span.slice_of(DOC) == "distributed"

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValidationError):
            Span(start=20, end=10)

    def test_rejects_empty_range(self) -> None:
        with pytest.raises(ValidationError):
            Span(start=10, end=10)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValidationError):
            Span(start=-1, end=5)

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((0, 10), (5, 15), True),
            ((0, 10), (10, 20), False),
            ((0, 20), (5, 10), True),
            ((0, 5), (10, 15), False),
        ],
    )
    def test_overlaps(self, a: tuple[int, int], b: tuple[int, int], expected: bool) -> None:
        assert Span(start=a[0], end=a[1]).overlaps(Span(start=b[0], end=b[1])) is expected


class TestCitation:
    def test_exact_quote_verifies(self) -> None:
        cite = Citation(
            document_id=DOC_ID,
            span=span_of("distributed task queue"),
            quote="distributed task queue",
        )
        assert cite.verify(DOC) is True

    def test_whitespace_and_case_differences_tolerated(self) -> None:
        cite = Citation(
            document_id=DOC_ID,
            span=span_of("distributed task queue"),
            quote="Distributed   Task  Queue",
        )
        assert cite.verify(DOC) is True

    def test_fabricated_quote_is_rejected(self) -> None:
        cite = Citation(
            document_id=DOC_ID,
            span=span_of("distributed task queue"),
            quote="led a team of fifteen engineers",
        )
        assert cite.verify(DOC) is False

    def test_out_of_bounds_span_is_rejected(self) -> None:
        cite = Citation(
            document_id=DOC_ID,
            span=Span(start=5000, end=5100),
            quote="anything at all",
        )
        assert cite.verify(DOC) is False

    def test_resolved_quote_comes_from_the_document_not_the_model(self) -> None:
        cite = Citation(document_id=DOC_ID, span=span_of("distributed"), quote="totally wrong")
        assert cite.resolved_quote(DOC) == "distributed"


class TestCited:
    def test_grounded_value(self) -> None:
        value: Cited[str] = Cited(
            value="Go",
            citations=[Citation(document_id=DOC_ID, span=span_of("Go"), quote="Go")],
        )
        assert value.is_grounded
        assert value.verify(DOC).ok

    def test_ungrounded_value_is_flagged(self) -> None:
        value: Cited[str] = Cited(value="Senior Engineer")
        assert not value.is_grounded
        assert value.verify(DOC).total == 0

    def test_inferred_constructor_records_lower_confidence(self) -> None:
        value = Cited.inferred("present", confidence=0.4)
        assert not value.is_grounded
        assert value.confidence == 0.4

    def test_bad_citation_surfaces_in_verification(self) -> None:
        value: Cited[str] = Cited(
            value="Rust",
            citations=[
                Citation(document_id=DOC_ID, span=span_of("Go"), quote="Go"),
                Citation(
                    document_id=DOC_ID,
                    span=span_of("distributed"),
                    quote="Rust systems work",
                ),
            ],
        )
        result = value.verify(DOC)
        assert result.total == 2
        assert result.valid == 1
        assert not result.ok
        assert result.rate == 0.5

    def test_generic_over_non_string_values(self) -> None:
        value: Cited[int] = Cited(
            value=40000,
            citations=[Citation(document_id=DOC_ID, span=span_of("40k"), quote="40k")],
        )
        assert value.value == 40000
        assert value.verify(DOC).ok


class TestVerificationResult:
    def test_rate_of_empty_result_is_one(self) -> None:
        assert VerificationResult(total=0, valid=0).rate == 1.0

    def test_results_fold(self) -> None:
        combined = VerificationResult(total=3, valid=3) + VerificationResult(
            total=2, valid=1, invalid_quotes=["made up"]
        )
        assert combined.total == 5
        assert combined.valid == 4
        assert combined.invalid_quotes == ["made up"]
        assert not combined.ok


class TestEvidenceUnit:
    def test_round_trips_into_a_citation(self) -> None:
        unit = EvidenceUnit(
            unit_id="u1",
            document_id=DOC_ID,
            text="distributed task queue",
            span=span_of("distributed task queue"),
            section="projects",
            page=1,
        )
        cite = unit.as_citation()
        assert cite.verify(DOC) is True
        assert cite.page == 1
