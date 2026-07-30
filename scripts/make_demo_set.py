"""Render a four-candidate demo set into ``samples/``.

Exists so the dashboard demo is one drag-and-drop rather than a hunt for files,
and so the shortlist screenshot shows a real spread. Four backend candidates
spanning strong, solid, mixed and weak means the ranked table demonstrates
ordering and at least one unmet must-have, instead of a single row that proves
nothing.

The resumes are synthetic, from the golden set. Putting a real person's
employment history in a public repository would be a privacy incident on a
project that argues for careful handling of candidate data.

Costs nothing: no API calls, no model, fully deterministic.
"""

from __future__ import annotations

from pathlib import Path

from hirelens.evals.golden import build_golden_set

# Filenames carry the candidate's name so the shortlist is readable in a
# screenshot. Blind mode redacts the name inside the document before the model
# sees it; the filename is metadata the model is never shown.
DEMO: dict[str, str] = {
    "c01": "alex_mercer.txt",
    "c02": "priya_raman.txt",
    "c03": "jordan_blake.txt",
    "c04": "sam_okafor.txt",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "samples" / "demo_candidates"
    out.mkdir(parents=True, exist_ok=True)

    golden = build_golden_set()
    by_id = {profile.candidate_id: profile for profile in golden.profiles}

    for candidate_id, filename in DEMO.items():
        profile = by_id[candidate_id]
        path = out / filename
        path.write_text(profile.render(), encoding="utf-8")
        print(f"{path.relative_to(root)}  ({profile.quality}, {len(profile.render())} chars)")

    backend = next(job for job in golden.jobs if job.job_id == "backend")
    jd = root / "samples" / "senior_backend_engineer_golden.txt"
    jd.write_text(backend.text, encoding="utf-8")
    print(f"{jd.relative_to(root)}  (the job description these four are graded against)")

    print(
        "\nDrag the four files in samples/demo_candidates onto the dashboard, "
        "and paste the job description above into the editor."
    )


if __name__ == "__main__":
    main()
