# Screenshots

The README links to five images from this folder. They are not committed until a
real run has produced them, because a screenshot of a run that has not happened
is a mockup, and the argument of this project is that a claim without evidence
behind it is worth nothing.

## The demo inputs

`make demo-set` renders four synthetic backend candidates as PDFs into
`samples/demo_candidates/`, plus the job description they are graded against:

| File | Golden-set quality | What it contributes to the shortlist |
|---|---|---|
| `alex_mercer.pdf` | strong | top of the ranking, most requirements met |
| `priya_raman.pdf` | solid | close behind, so the intervals overlap |
| `jordan_blake.pdf` | mixed | middle, with disagreement across samples |
| `sam_okafor.pdf` | weak | bottom, and misses a must-have |

Synthetic on purpose. A real person's employment history in a public repository
would be a privacy incident on a project that argues for careful handling of
candidate data.

## Order matters, because the free tier has a daily cap

Take them in this order. It is not arbitrary.

1. **`audit.png` first.** Needs no API key at all, so it can never be blocked.
2. **`cli.png` second**, on one candidate. About 32 calls. If anything in the
   pipeline is broken, you find out cheaply rather than 125 calls in.
3. **`shortlist.png`, `evidence.png`, `progress.png` last**, from one dashboard
   run over all four candidates. Roughly 94 further calls, because candidate one
   is already in the response cache from step 2. Use the **same** job description
   in both steps or that saving disappears.

***

## 1. `audit.png`: the bias audit catching injected bias

```powershell
python scripts/smoke_audit.py
```

Capture the second case, the one rigged to reward elite institutions, with both
tables visible: blind mode clean, sighted mode over threshold. It proves the
instrument detects bias that is definitely there, which is the only thing that
makes a clean report on the real model meaningful.

## 2. `cli.png`: the terminal report

```powershell
hirelens score samples\demo_candidates\alex_mercer.pdf --jd samples\senior_backend_engineer_golden.txt
```

Capture the whole report: score with band, per-requirement verdicts with the
evidence quoted underneath, risk flags, interview questions. Some readers never
open the interface.

## 3. `shortlist.png`: the ranked table

Screen all four candidates against the same job description, then capture the
shortlist with the confidence bars visible.

What this has to show: several candidates, **at least one confidence interval
overlapping another**, and at least one candidate marked as missing a must-have.
The overlap is the point. A table where every candidate is cleanly separated
hides the honest argument.

## 4. `evidence.png`: evidence highlighting

Open a candidate, select one requirement, and capture both panes together: the
requirement selected on the left, its evidence highlighted and everything else
dimmed on the right.

This is the one screenshot that has to be right. It is the difference between
this project and every other resume screener on GitHub.

## 5. `progress.png`: a run in flight

Capture while the progress bar is partway through, with the stage label visible
("judging requirements", "extracting", and so on). At the paced request rate the
run takes minutes, so there is plenty of time. Take a burst and pick one.

***

## What to avoid

- **No API keys in frame.** Check the terminal scrollback before capturing, and
  make sure the browser network tab is closed.
- **No cropped confidence intervals.** If the bands are cut off, the screenshot
  shows a ranking and hides the uncertainty, which inverts the message.
- **No `.env` file open in another visible window.**

Set the browser to roughly 1440 x 900 and crop to the page content, without the
URL bar or bookmarks.

## Paste this into the README

Replace the HTML comment under "The dashboard" with:

```markdown
| | |
|---|---|
| ![Ranked shortlist](docs/screenshots/shortlist.png) | ![Evidence highlighting](docs/screenshots/evidence.png) |
| Ranked shortlist. Overlapping intervals mean the ordering between those two candidates is not meaningful. | Selecting a requirement highlights the exact characters that produced its verdict. |
```

And under "Commands", after the terminal example:

```markdown
![CLI output](docs/screenshots/cli.png)
```
