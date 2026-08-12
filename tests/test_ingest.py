from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from hirelens.ingest import IngestionError, SourceFormat, read_document
from hirelens.ingest.document import BoundingBox, SourceDocument, TextAccumulator

RESUME = """ADWITIYA SHUKLA
adwitiya@example.com | github.com/adwitiya

EXPERIENCE
Backend Engineer, Acme Corp (2023 - present)
Cut p99 checkout latency from 1.2s to 180ms by replacing the ORM hot path.
Owned the payments reconciliation service handling 40k transactions daily.

PROJECTS
hirelens - Evidence-grounded resume screening with a bias audit harness.

SKILLS
Python, Go, PostgreSQL, Kubernetes, PyTorch
"""


@pytest.fixture
def resume_txt(tmp_path: Path) -> Path:
    path = tmp_path / "resume.txt"
    path.write_text(RESUME, encoding="utf-8")
    return path


class TestTextIngestion:
    def test_reads_a_text_file(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        assert doc.source_format is SourceFormat.TEXT
        assert doc.filename == "resume.txt"
        assert "Kubernetes" in doc.text

    def test_every_block_span_slices_back_to_its_own_text(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        assert doc.blocks, "expected at least one indexed block"
        for block in doc.blocks:
            assert doc.slice(block.span) == doc.text[block.span.start : block.span.end]
            assert doc.slice(block.span).strip() != ""

    def test_blocks_are_in_document_order(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        starts = [b.span.start for b in doc.blocks]
        assert starts == sorted(starts)

    def test_blocks_do_not_overlap(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        for earlier, later in pairwise(doc.blocks):
            assert earlier.span.end <= later.span.start

    def test_all_caps_section_headers_are_detected(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        headings = {doc.slice(b.span).strip() for b in doc.blocks if b.is_heading}
        assert {"EXPERIENCE", "PROJECTS", "SKILLS"} <= headings

    def test_document_id_is_content_addressed(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text(RESUME, encoding="utf-8")
        b.write_text(RESUME, encoding="utf-8")
        assert read_document(a).document_id == read_document(b).document_id

    def test_different_content_gives_a_different_id(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text(RESUME, encoding="utf-8")
        b.write_text(RESUME + "\nExtra line", encoding="utf-8")
        assert read_document(a).document_id != read_document(b).document_id

    def test_line_at_returns_the_containing_line(self, resume_txt: Path) -> None:
        doc = read_document(resume_txt)
        offset = doc.text.index("Kubernetes")
        assert "Python, Go, PostgreSQL, Kubernetes, PyTorch" in doc.line_at(offset)


class TestErrorHandling:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not found"):
            read_document(tmp_path / "nope.pdf")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(IngestionError, match="empty"):
            read_document(path)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "resume.rtf"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(IngestionError, match="Unsupported file type"):
            read_document(path)


class TestTextAccumulator:
    def test_blank_lines_advance_the_offset_without_indexing_a_block(self) -> None:
        acc = TextAccumulator()
        acc.add_line("first", page=1)
        assert acc.add_line("   ", page=1) is None
        acc.add_line("second", page=1)

        doc = acc.build(
            document_id="x",
            filename="x.txt",
            source_format=SourceFormat.TEXT,
            page_count=1,
        )
        assert len(doc.blocks) == 2
        assert doc.slice(doc.blocks[1].span) == "second"

    def test_bounding_boxes_are_preserved_for_highlighting(self) -> None:
        acc = TextAccumulator()
        acc.add_line("Backend Engineer", page=2, bbox=BoundingBox(x0=72, y0=100, x1=300, y1=118))
        doc = acc.build(
            document_id="x", filename="x.pdf", source_format=SourceFormat.PDF, page_count=2
        )

        boxes = doc.highlight_boxes(doc.blocks[0].span)
        assert boxes == [(2, BoundingBox(x0=72, y0=100, x1=300, y1=118))]
        assert doc.page_of(doc.blocks[0].span) == 2

    def test_scanned_pdf_heuristic(self) -> None:
        acc = TextAccumulator()
        acc.add_line("Resume", page=1)
        doc = acc.build(
            document_id="x", filename="scan.pdf", source_format=SourceFormat.PDF, page_count=3
        )
        assert doc.is_probably_scanned


class TestBoundingBox:
    def test_union_covers_both(self) -> None:
        merged = BoundingBox(x0=0, y0=0, x1=10, y1=10).union(BoundingBox(x0=5, y0=5, x1=20, y1=20))
        assert (merged.x0, merged.y0, merged.x1, merged.y1) == (0, 0, 20, 20)
        assert merged.width == 20


class TestSourceDocumentIds:
    def test_make_id_is_stable_and_short(self) -> None:
        first = SourceDocument.make_id(b"payload")
        assert first == SourceDocument.make_id(b"payload")
        assert len(first) == 16
