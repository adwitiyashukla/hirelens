from __future__ import annotations

from pathlib import Path

import fitz

from hirelens.evals.golden import build_golden_set

DEMO: dict[str, str] = {
    "c01": "alex_mercer",
    "c02": "priya_raman",
    "c03": "jordan_blake",
    "c04": "sam_okafor",
}

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 56.0, 64.0, 56.0
LINE_H = 13.5

SECTIONS = {"EXPERIENCE", "PROJECTS", "EDUCATION", "SKILLS", "AWARDS"}


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
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
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN_TOP
    usable = PAGE_W - 2 * MARGIN_X

    for index, raw in enumerate(resume_text.splitlines()):
        line = raw.rstrip()

        if not line:
            y += LINE_H * 0.55
            continue

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
