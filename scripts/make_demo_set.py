"""Render a four-candidate demo set into ``samples/demo_candidates``.

Exists so the dashboard demo is one drag-and-drop rather than a hunt for files,
and so the shortlist screenshot shows a real spread. Four backend candidates
spanning strong, solid, mixed and weak means the ranked table demonstrates
ordering and at least one unmet must-have, instead of a single row that proves
nothing.

Writes PDFs, not just text. Text would work, but a PDF exercises the part of the
pipeline that is actually hard: rebuilding reading order from PyMuPDF's geometry,
then mapping character offsets back to a page so a citation can say "page 1" and
mean it. A demo run against .txt inputs would quietly skip all of that and show
citations with no page anchor.

The resumes are synthetic, from the golden set. Putting a real person's
employment history in a public repository would be a privacy incident on a
project that argues for careful handling of candidate data.

Costs nothing: no API calls, no model, fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from hirelens.evals.golden import build_golden_set

# Filenames carry the candidate's name so the shortlist is readable in a
# screenshot. Blind mode redacts the name inside the document before the model
# sees it; the filename is metadata the model is never shown.
DEMO: dict[str, str] = {
    "c01": "alex_mercer",
    "c02": "priya_raman",
    "c03": "jordan_blake",
    "c04": "sam_okafor",
}

# A4 in points, with margins wide enough that no line reaches the edge. PyMuPDF
# clips text that overflows the page rather than wrapping it, so wrapping is done
# here explicitly.
PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 56.0, 64.0, 56.0
LINE_H = 13.5

SECTIONS = {"EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS", "AWARDS"}


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy wrap using real glyph widths, not a character count.

    A character-count wrap overflows on capitals and underflows on lowercase, and
    the overflow is the half that gets silently clipped off the page.
    """
    if not text:
        return [""]

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if fitz.get_text_length(candidate, fontname=font, fontsize=size) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    lines.append(current)
    return lines


def render_pdf(resume_text: str, out: Path) -> int:
    """Write a plain, readable resume PDF. Returns the page count."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN_TOP
    usable = PAGE_W - 2 * MARGIN_X

    for index, raw in enumerate(resume_text.splitlines()):
        line = raw.rstrip()

        if not line:
            y += LINE_H * 0.55
            continue

        # Line 0 is the name. Section headings are the all-caps words we know by
        # name. Everything else is body text.
        if index == 0:
            font, size = "helvetica-bold", 15.0
        elif line.strip() in SECTIONS:
            font, size = "helvetica-bold", 10.5
            y += LINE_H * 0.45
        else:
            font, size = "helvetica", 9.8

        for piece in _wrap(line, font, size, usable):
            if y > PAGE_H - MARGIN_BOTTOM:
                page = doc.new_page(width=PAGE_W, height=PAGE_H)
                y = MARGIN_TOP
            page.insert_text((MARGIN_X, y), piece, fontname=font, fontsize=size)
            y += LINE_H if size < 12 else LINE_H * 1.5

    doc.save(out, deflate=True)
    pages = doc.page_count
    doc.close()
    return pages


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "samples" / "demo_candidates"
    out.mkdir(parents=True, exist_ok=True)

    golden = build_golden_set()
    by_id = {profile.candidate_id: profile for profile in golden.profiles}

    for candidate_id, stem in DEMO.items():
        profile = by_id[candidate_id]
        text = profile.render()

        # Both formats. The PDF is what the demo uses; the text file is there so
        # the exact input stays readable in a diff without opening a viewer.
        (out / f"{stem}.txt").write_text(text, encoding="utf-8")
        pages = render_pdf(text, out / f"{stem}.pdf")

        print(f"{stem}.pdf  {pages} page(s), {len(text):>5} chars, quality={profile.quality}")

    backend = next(job for job in golden.jobs if job.job_id == "backend")
    jd = root / "samples" / "senior_backend_engineer_golden.txt"
    jd.write_text(backend.text, encoding="utf-8")
    print(f"\n{jd.relative_to(root)}\n  the job description these four are graded against")

    print(
        "\nDrag the four PDFs in samples/demo_candidates onto the dashboard, "
        "and paste the job description above into the editor."
    )


if __name__ == "__main__":
    main()
