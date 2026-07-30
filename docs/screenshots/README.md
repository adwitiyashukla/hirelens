# Screenshots

The README links to five images from this folder. They are not committed yet, because a
screenshot of a run that has not happened would be a mockup, and the whole argument of this
project is that a claim without evidence behind it is worth nothing.

Take them once, after a real run against a live provider, then paste the markdown block at the
bottom of this file into the README where the HTML comment marks the spot.

## Before you start

```bash
make web-build                 # compile the dashboard
make api                       # serve it on http://localhost:8000
```

Open `http://localhost:8000`. Set your browser window to roughly 1440 x 900 and use a light-free
desktop background, so the dark interface is the only thing in the frame. Crop to the browser
content area, without the URL bar or bookmarks.

## The five captures

### 1. `shortlist.png`: the ranked table

Screen three or four candidates against one job description. Capture the shortlist with the
confidence bars visible.

What this has to show: several candidates, at least one confidence interval overlapping another,
and at least one candidate marked as missing a must-have. Overlapping intervals are the point.
A table where every candidate is cleanly separated makes the honest argument invisible.

### 2. `evidence.png`: evidence highlighting

Open a candidate, select one requirement, and capture both panes together: the requirement
selected on the left, its evidence highlighted and everything else dimmed on the right.

This is the one screenshot that has to be right. It is the difference between this project and
every other resume screener on GitHub.

### 3. `progress.png`: a run in flight

Start a run and capture while the progress bar is partway through, with the stage label visible
("judging requirements", "extracting", and so on). Timing it is fiddly; taking a burst of
screenshots and picking one is easier than trying to catch it.

### 4. `cli.png`: the terminal output

```bash
hirelens score samples\priya_narayanan.pdf --jd samples\senior_backend_engineer.txt
```

Capture the whole report: score with band, per-requirement verdicts with evidence, risk flags,
interview questions. Some recruiters read the terminal output and skip the interface entirely.

### 5. `audit.png`: the bias audit catching injected bias

```bash
make audit-smoke
```

Capture the second case, the one rigged to reward elite institutions, with both tables visible:
blind mode clean, sighted mode over threshold. This needs no API key, so it can be taken at any
time.

## What to avoid

- **No real resumes.** Use `samples/` or the golden set. A screenshot of a real person's
  employment history in a public repository is a privacy incident, and on a project that argues
  for careful handling of candidate data it would undermine the entire premise.
- **No API keys in frame.** Check the terminal scrollback before capturing, and check the
  browser's network tab is closed.
- **No cropped confidence intervals.** If the bands are cut off, the screenshot shows a ranking
  and hides the uncertainty, which inverts the message.

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
