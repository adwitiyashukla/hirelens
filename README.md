# HireLens

A resume screener where every score points back at the line of the resume it came from.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Runs free](https://img.shields.io/badge/API%20cost-%240.00-brightgreen.svg)](#running-it)

## Why I did not build the obvious version

The obvious version of this project is about fifteen lines:

```
resume.pdf  ->  extract text  ->  "score this candidate out of 100"  ->  87
```

I wrote that in an afternoon and then sat looking at the number 87 trying to work out what I was
supposed to do with it.

It cannot explain itself. There is no line of the resume I can point at and say this is where the
87 came from. It is not reproducible either: I ran it twice on the same PDF and got 87, then 81,
and nothing in the tool would ever have told me that. And it has never been checked against a
person, so there is no way to say whether the ranking is any good.

In a lot of places it would also be illegal to ship. NYC Local Law 144 requires an annual bias
audit of any automated employment decision tool, and the EU AI Act puts employment screening in
its high-risk category.

So I built the other thing. The rule I gave myself was that a screening decision is only worth
anything if I can defend it, and every choice below falls out of that.

| Property | What it means | How it is built |
|---|---|---|
| Grounded | Every claim points to an exact character range in the source document | Span-tracked ingestion, citation-carrying schemas, retrieval-scoped judging |
| Stable | Re-running the same input gives the same decision, and the spread is published | Self-consistency sampling, median aggregation, bootstrap intervals |
| Measured | There is a number for how well it agrees with a human screener | Golden set, Spearman correlation, three baselines, a CI regression gate |
| Fair | Demographic swaps are actively tested and the build fails if they move the score | Counterfactual perturbation harness with a null control |

## Where the data came from

There is no public dataset of resumes with expert screening labels attached, and for good reason.
Real resumes are personal data and I was not going to scrape them.

So the golden set is synthetic, and that is the weakest part of the project. It is 12 candidate
profiles across 3 job descriptions, giving 36 candidate-job pairs. Each profile is a structured
spec plus a deterministic renderer rather than a PDF I wrote by hand, which turned out to matter
more than I expected. The profiles are diffable in git, they render identically every time so the
response cache hits and runs are reproducible, and demographic attributes live in their own
fields. That last part is what makes the bias audit possible at all: swap only the name and the
university, re-render, re-score, and any movement was caused by the swap rather than correlated
with it. A test asserts that everything outside the demographic block is byte-identical between
variants.

## The data problem that took real thought

PDFs do not have lines. They have text boxes with coordinates, handed to you in whatever order the
generator happened to write them. On a two-column resume the skills sidebar gets interleaved with
the work history, and any character offset you compute from that is meaningless.

Which matters, because the whole project rests on offsets. A citation is a character range. If the
ranges are wrong, every highlight in the dashboard points at the wrong text.

So ingestion rebuilds lines from the geometry: group spans by vertical position, find column
boundaries from the horizontal gaps, emit text in reading order, and keep a map from every output
character back to the block and bounding box it came from. That map is what lets the dashboard
draw a rectangle on the original PDF later.

Redaction is the other half of the same problem. Blind mode strips names, emails, institutions and
locations before the model sees anything, but if masking changes the length of the text then every
offset computed before it is wrong. So masks are length-preserving. `PRIYA NARAYANAN` becomes
`[NAME]#########`, same character count, 639 characters in and 639 out. Redaction stores spans
rather than rewriting the document, because the fairness harness needs to know exactly which
characters to perturb.

## How the pipeline works

```
                        +------------------------------------------+
   job description ---->|  RUBRIC COMPILER                         |
                        |  JD -> weighted atomic requirements      |
                        +--------------------+---------------------+
                                             |  Requirement[]
                                             v
   resume.pdf --> INGEST --> REDACT --> EXTRACT --> INDEX --> MATCH --> JUDGE --> AGGREGATE
                    |          |           |          |         |         |          |
              offset-preserving|      cited fields  chunk +   hybrid   per-req   weighted
              text + layout    |      (span-linked) embed   retrieval  scoring    rollup
                               |                                                     |
                          PII spans                                                  v
                                                                        +----------------------+
   github user --> ENRICH (async, cached) ------------------------->    |  ASSESSMENT          |
                                                                        |  score + spread      |
                                                                        |  citations           |
                                                                        |  risk flags          |
                                                                        |  interview pack      |
                                                                        +----------+-----------+
                                                                                   |
                        +----------------------------------------------------------+
                        v                                                          v
              +-------------------+                                +-----------------------+
              |  EVAL HARNESS     |                                |  FAIRNESS AUDIT       |
              |  vs golden set    |                                |  counterfactual probe |
              |  rho, MAE, sigma  |                                |  score drift report   |
              +-------------------+                                +-----------------------+
```

Three decisions do most of the work.

### The schema makes ungrounded output a validation error

Every extracted value is a `Cited[T]`, a value plus the character span it came from. There is no
way to record a company name without also recording where it appeared.

```python
class WorkExperience(BaseModel):
    company: Cited[str]
    highlights: list[Cited[str]]
```

A citation whose offsets do not contain the quoted text fails verification, so a hallucination
becomes a countable event in a report instead of a quiet lie in the output.

I never ask the model for the offsets. Language models cannot count characters, and a model asked
for a span returns confident wrong integers. It reports the verbatim text it read, and a string
search finds where that text lives: exact match first, then whitespace-normalised, then fuzzy.
That split is why citation validity comes out in the high nineties instead of being something I
just hoped for.

### The judge never sees the whole resume

For each requirement I retrieve only the handful of resume lines relevant to it, then ask the
model to judge that one requirement against that one piece of evidence. The prompts are small,
they run in parallel, and there is not enough context in the window for the model to borrow an
achievement from elsewhere in the document.

Retrieval is BM25 and dense embeddings fused with reciprocal rank fusion, and every hit records
which ranker found it. I wrote the BM25 in-repo because it is about forty lines and the corpus is
a single resume, so an industrial implementation buys a dependency and nothing else. Retrieval
favours recall on purpose: a requirement the candidate clearly does not meet still returns its
least-irrelevant chunks, because thresholding an uncalibrated fusion score would sometimes hide
the one piece of evidence that mattered. The judge is allowed to answer that none of this
supports the requirement.

### The judge picks a label, not a number

Asking a model to score 0 to 100 gives badly calibrated output. It clusters on round numbers,
returns 75 and 80 for evidence you cannot tell apart, and drifts when you reword the prompt. So it
picks one of five anchored verdicts (`strong`, `clear`, `partial`, `weak`, `none`) and I apply the
numeric coefficient myself. Hiring panels use anchored scales for the same reason.

Each requirement is judged `k=5` times at temperature 0.3. The median is the score and the sample
spread is the confidence band. Median rather than mode, because at k=5 the mode is decided by a
single draw on a 2-2-1 split. Median rather than mean, because averaging labels means treating
them as numbers again, which is the problem the ordinal scale exists to avoid. Even splits resolve
downward, since overstating a candidate is the more expensive mistake for whoever reads the report.

One thing I got wrong on the first attempt: each of the k samples needs a distinct nonce in its
request. Without one, all five calls hash to the same cache entry, return the same verdict five
times, and the band is zero every time. That is the easiest way to accidentally fake
self-consistency, and I only caught it because a band of exactly 0.0 on every requirement looked
too good.

## The dashboard

The screening view is deliberately not a leaderboard. It is an evidence viewer that happens to be
ranked. Selecting a requirement highlights the exact characters that produced its verdict, and
clicking a highlight selects the requirement it belongs to. That round trip is the whole argument,
because a number you can trace back to a line of text is a different object from a number a model
produced.

![Screening run in progress](docs/screenshots/dashboard_run.png)

A run in flight against `llama-3.3-70b-versatile`. Page and character counts are read back from
the offset map that citations get resolved against later, and progress arrives over server-sent
events, so the stage label is the pipeline's real position rather than a spinner. The green chip
is not decoration: names, emails and institutions are already masked in what the model is reading.

![Compiled rubric](docs/screenshots/dashboard_rubric.png)

The same run, finished. The job description was compiled into weighted atomic requirements before
any resume was read, which is what makes scores comparable across candidates. One candidate could
not be assessed and the run says so rather than scoring them zero. A quota failure is not evidence
about a candidate, and that banner exists because an earlier version of this system did exactly
that.

I have not captured the ranked shortlist or the evidence highlighter yet. Both need a run where
every candidate scores cleanly and free-tier quotas made that slow to arrange. Nothing here is
mocked.

What the interface refuses to do is as deliberate as what it does:

- Scores never appear without their interval, so two candidates whose bands visibly overlap do not have a meaningful ordering.
- Requirements whose repeated samples disagreed get labelled rather than averaged into a clean-looking number.
- A candidate missing a must-have ranks below everyone who meets them, whatever the totals say. A 75 that fails a hard requirement is not better than a 62 that meets everything, and one number cannot express that.
- Citations are re-checked against the stored document on page load rather than trusted, and any that fail are highlighted in red.
- Failed candidates are excluded rather than scored zero, and the failure count is reported separately.

```bash
make web-install   # once
make demo          # builds the dashboard, serves everything on http://localhost:8000
```

`make api` and `make web` in two terminals gives hot reload on both sides, with Vite proxying
`/api` so there is no CORS to configure.

## Three bugs worth reading about

I developed the whole pipeline against Gemini. Then I pointed it at Groq as a second provider and
it broke in three different ways in one afternoon, and a fourth turned up while I was fixing the
third. All of them were silent: the system produced confident, well-formed, completely wrong
output while every quality metric in the report read 100%.

### The schema was never sent to the provider that needed it most

Gemini accepts a JSON schema with the request. Groq's API offers only "reply with valid JSON" and
no schema at all, and I had not noticed. So the model was inferring field names from the prose in
my prompt, getting them wrong, and the repair loop was burning its attempts on the same mistake.

It eventually settled on a sparse object that validated, because the fields it had missed all had
defaults. The rubric compiled to eight requirements with zero must-haves, so every candidate
trivially met all of them, and a genuinely strong candidate scored 0 out of 100.

The fix renders the schema into the prompt as a field list with enum values when the provider
cannot take a real one. What I actually learned is that adding a default to make validation pass
turns a loud failure into a silent one.

### Grounding and citation validity are ratios, so they were perfect over an empty set

That 0-out-of-100 report showed grounding 100%, citations valid 100%, agreement 100%. All three
were true. One claim had been extracted and it was correctly cited, so every ratio was 1 over 1.

Then it clicked that these metrics only measure internal consistency. They say nothing about
coverage, and a system that extracts nothing scores perfectly on all of them.

Two absolute floors sit alongside the ratios now. Extraction that yields almost nothing from a
substantial document raises instead of scoring, and a rubric that compiles to no must-haves from a
posting that clearly lists requirements gets one corrective retry before being rejected.

### The rate limiter counted the wrong thing

I was pacing at a comfortable 25 requests a minute and getting rate limited anyway, which made no
sense until I read the Groq docs properly. Groq meters tokens per minute, not requests. Each call
carried roughly a thousand tokens, so 25 requests a minute was aiming at 25,000 TPM against a
12,000 ceiling.

The visible symptom was much worse than a 429. Extraction calls exhausted their retries, the
resume came back with almost no evidence attached, and the candidate was reported as a weak match.
A quota shortfall had turned itself into a hiring signal, which is exactly the failure class this
project exists to prevent. The limiter now runs two leaky buckets, one for requests and one for
tokens, and reserves token budget in proportion to each request's estimated size.

### And a fourth, found while fixing the third

Responses that failed schema validation were being left in the cache. Since prompts are
deterministic, the next run replayed the same malformed answer and failed identically, forever,
long after the provider itself was fine. Invalid responses get evicted now. The same investigation
turned up that the test suite was reading my local `.env`, so adding one line to it made all 397
tests hang. Tests are isolated by an autouse fixture now.

## Evaluation

Most resume screeners are never evaluated at all. This one is built around the harness, and it is
the part I would point an interviewer at first.

```bash
make golden      # render the 12-profile golden set so you can read it
make label       # assign human screening tiers, the one step a model cannot do
make eval        # run the harness and print the metrics table
make eval-smoke  # exercise the harness with no API key at all
```

| Metric | Question it answers |
|---|---|
| Spearman rho, with a bootstrap 95% interval | Do we order candidates the way a person would? |
| Kendall tau-b | Same question, less sensitive to one badly misplaced candidate |
| Pairwise inversion rate | What share of head-to-head comparisons do we get backwards? |
| Top-3 precision | Is the top of the shortlist right, which is all a recruiter reads? |
| Citation validity | Do the cited spans really contain the quoted text? |

Three baselines run on the same 36 pairs: random ordering, keyword overlap, and BM25 using the job
description as the query. A correlation with nothing to compare it against is not evidence. If
HireLens cannot beat plain BM25 then everything I built on top of retrieval is not earning its
complexity, and the gate fails the build rather than letting that slide. The harness also prints a
chance ceiling, the correlation random ordering reaches 5% of the time on a set this size. On 12
candidates that is about 0.29, not zero, so a bare rho quoted without it means very little.

Metrics are implemented in-repo rather than pulled from scipy. Ties are the normal case with
tiered human labels and the tie corrections are the part worth being explicit about.

### What the numbers are, and what they are not

The machinery runs on every commit. The human labels do not exist yet, and I am not publishing a
correlation until they do. `make eval-smoke` swaps in a keyword-matching stand-in for the model
plus placeholder labels, and confirms all 36 pairs flow through every stage:

```
ranker                     spearman   kendall   inversions   top-3
--------------------------------------------------------------------------
HireLens         0.669 [0.41, 0.82]     0.486       23.7%     67%
bm25             0.590 [0.27, 0.80]     0.456       26.0%    100%
keyword          0.541 [0.20, 0.77]     0.408       27.4%    100%
random          0.029 [-0.32, 0.38]     0.028       45.4%      0%
--------------------------------------------------------------------------
random 95th                   0.288   (chance ceiling)

citation validity  100.0%   grounding rate  100.0%   self-consistency  100.0%
```

Read that as plumbing, not quality. The labels are placeholders, so 0.669 is a number about a
keyword matcher agreeing with a heuristic, not about HireLens agreeing with a person. The script
says so in its own output and I am repeating it because a table like that is easy to quote out of
context.

The interval is worth looking at even so. On 36 pairs a rho of 0.669 carries a 95% interval of
roughly 0.41 to 0.82. Quoting the point estimate alone would be overclaiming by a wide margin, and
the width of that interval is itself the finding: 36 pairs is not enough. It is also why every
point estimate in this project ships with a bootstrap interval.

## Bias audit

NYC Local Law 144 requires an annual bias audit of automated employment decision tools, and the EU
AI Act classifies employment screening as high-risk. This is the measurement such a filing would
be built on.

```bash
make audit-plan   # show the experiment matrix and exact cost, spending nothing
make audit        # run it
make audit-smoke  # prove the audit catches injected bias, no API key needed
```

The method is the audit-study design from labour economics, pointed at a model instead of at
employers. Bertrand and Mullainathan sent out identical resumes varying only the name and measured
callback rates. Here the resume is held byte-identical apart from the demographic block and the
outcome is the score. Four axes: gender-coded names, ethnicity-coded names, university prestige,
and location.

Two design choices carry the whole thing.

The first is a null control. The same unmodified resume is scored twice, and that difference is
the system's own run-to-run noise. Every axis is then reported as excess drift above that floor. A
three-point swing when the name changes means nothing if the score swings three points when
nothing changes at all. It is also why the audit turns the response cache off: with caching on,
two identical runs hit the same cache entry and the noise floor is zero by construction rather
than by measurement. A test asserts the cache is off even when a client is injected.

The second is measuring blind and sighted both. Blind mode is how the system ships, so blind
results gate the build. The sighted run measures the underlying model bias that masking
suppresses, and the difference between the two says what the mitigation is actually worth. The
gate never fails on sighted numbers, because those exist to measure the problem blind mode solves,
not to punish me for having measured it.

### Testing the audit against a model I rigged to be biased

An audit that reports clean on a definitely-biased system is worse than no audit, so
`make audit-smoke` runs two cases against a stand-in model. One is clean. The other is rigged to
upgrade its verdict whenever an elite institution appears.

```
CASE 2: stand-in rigged to reward elite institutions

BLIND MODE ON (shipping configuration)
axis                           max   excess     gap  flips  verdict
university prestige           0.00     0.00    0.00      0  ok

BLIND MODE OFF (diagnostic)
axis                           max   excess     gap  flips  verdict
university prestige          63.40    63.40   24.08      3  OVER THRESHOLD

blind mode removes 63.40 pts of worst-case drift (63.40 -> 0.00)
largest sighted gap: 'elite' scores 24.08 pts above 'mid' on the university prestige axis
```

The audit sees the injected bias, attributes it to the right axis and the right group, counts the
three shortlist positions it moved, and confirms masking blocks it completely. Both cases run in
CI on every commit.

| | |
|---|---|
| ![Audit on a clean model](docs/screenshots/audit_case1.png) | ![Audit catching injected bias](docs/screenshots/audit_case2.png) |
| **Case 1, clean stand-in.** Every axis at 0.00 drift, no flips, audit passes. An instrument that cannot stay quiet on a clean system is useless. | **Case 2, rigged stand-in.** 63.40 points of drift with blind mode off, 0.00 with it on, and 3 shortlist positions moved. |

The pair is the point. Case 2 alone would only show the audit fires, Case 1 alone would only show
it stays quiet. Together they show it fires when and only when there is something to find. These
are stand-in numbers proving the instrument works, not a result about a real model. Running it
against a real provider is one command and writes `docs/BIAS_AUDIT.md`.

One rule falls out of all this and applies to the main pipeline too: risk flags never change the
score. An employment gap can be caregiving, illness, study, or a startup that failed, and every one
of those correlates with a protected characteristic. Deducting points would encode the exact bias
this audit exists to detect. The flag is surfaced and a human decides.

## Running it

No paid API is required. The embedding model runs locally on CPU and all three backends have a
free path.

```
git clone https://github.com/adwitiyashukla/hirelens
cd hirelens

python -m venv .venv
.venv\Scripts\activate             # macOS or Linux: source .venv/bin/activate
pip install -e ".[dev]"

copy .env.example .env             # macOS or Linux: cp .env.example .env
```

Then pick one backend and put the key in `.env`:

| Backend | Key from | Notes |
|---|---|---|
| Gemini | [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys) | Free tier, generous limits, a minute to set up |
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | Free tier, very fast |
| Ollama | no key needed | `ollama pull qwen3:4b`. Fully local and offline. |

I kept Ollama working even after I had keys for the other two. It is the only backend someone can
run with zero signup, and it makes the provider comparison a three-way result instead of two-way.

```bash
hirelens doctor                            # check config and provider connectivity
hirelens models                            # list the models your key can actually call
hirelens ingest resume.pdf --blocks 15     # inspect the offset map citations are built on
hirelens redact-preview resume.pdf         # see exactly what blind mode removes
hirelens parse resume.pdf --json out.json  # structured resume, every field cited
hirelens match resume.pdf job.txt          # which resume lines answer which requirement
hirelens score *.pdf --jd job.txt          # ranked shortlist with evidence and bands
```

`score` against a senior backend job description. Note the confidence band, the per-requirement
agreement, and the one requirement flagged for review because repeated runs disagreed:

```
ROLE      : Senior Backend Engineer (senior)
CANDIDATE : candidate-87c7a5a2
SCORE     : 81/100 (band 81 to 83), strong fit, agreement 93%
MUST-HAVES: all met
QUALITY   : grounding 95%, citations valid 100%, 15 evidence units

[MUST] Has run containerised workloads in production   clear    20.0/25.0  agree 100%
       evidence: p1 Kubernetes
[MUST] Has worked with high-throughput event streaming strong   25.0/25.0  agree 100%
       evidence: p1 Built a Kafka consumer group processing 2M settlement...
[MUST] Has measurably improved system performance      strong   25.0/25.0  agree 100%
       evidence: p1 Cut p99 checkout latency from 1.2s to 180ms by remov...
[nice] Has contributed to open source                  partial   4.2/ 8.3  agree 60%  <- NEEDS REVIEW
       evidence: p1 Semantic diff for Jupyter notebooks. Merged upstream...
[nice] Has infrastructure as code experience           none      0.0/ 8.3  agree 100%

RISK FLAGS
  [medium] 2 project(s) have no repository or demo link. Their claims cannot be checked.
  [medium] 1 requirement(s) produced inconsistent verdicts across repeated runs.

INTERVIEW QUESTIONS
  1. Walk me through how the Kafka consumer group handles rebalances and duplicate deliveries.
     -> Establishes depth behind the 2M events/day claim.
  2. What was your specific contribution to the nbdime upstream merge?
     -> Open source evidence was borderline across runs.
```

`redact-preview` on a sample resume. Identity goes, every technical claim and every character
offset survives:

```
category      found   examples
-----------   -----   ---------------------
email             1   priya.n@example.com
institution       1   NIT Trichy
location          1   Bengaluru
name              1   PRIYA NARAYANAN
url               1   github.com/pnarayanan

offsets preserved: yes (639 chars in, 639 out)

[NAME]#########
[EMAIL]############  |  github.com/pnarayanan  |  #########
Cut p99 checkout latency from 1.2s to 180ms by removing an ORM N+1 in the
```

## The HTTP API

```bash
pip install -e ".[api]"
make api          # http://localhost:8000/docs, SQLite, no Docker needed
make up           # or the same thing on Postgres in Docker
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Post a job description. Idempotent on the text. |
| `POST` | `/api/documents` | Upload resumes. Idempotent on file bytes, partial success on bad files. |
| `POST` | `/api/runs` | Start screening. Returns 202 immediately. |
| `GET` | `/api/runs/{id}/events` | Live progress over server-sent events. |
| `GET` | `/api/runs/{id}/shortlist` | Ranked candidates, must-haves ahead of score. |
| `GET` | `/api/assessments/{id}` | One candidate with citations and highlight rectangles. |
| `GET` | `/api/documents/{id}/text` | Canonical text plus the offset map, for highlighting. |

Uploads and job posts are content-addressed, so an ID is a hash of the bytes rather than a fresh
UUID. The same resume uploaded twice resolves to the same record and reuses its cached extraction,
and the response says which of the two happened instead of silently creating a duplicate
candidate. For job descriptions that is a correctness requirement rather than a convenience,
because recompiling the rubric would make two runs incomparable.

A bad file does not fail the batch: thirty resumes with one corrupt PDF returns twenty-nine
accepted and one rejected with a reason. Citations are re-verified at read time rather than
trusted from the stored payload, and each carries the rectangles needed to draw over the PDF.

## Deploying it

The image is multi-stage. Node builds the dashboard, Python serves both it and the API from one
process. One container, no CORS, no separate frontend host, which is what the free tiers give you.

```bash
docker compose up --build     # locally, with Postgres
```

`render.yaml` is a Render blueprint and `fly.toml` is a Fly config. Both set the model key as a
secret rather than an environment value in the repository, and both pace requests well under the
free-tier quota, because a 429 in front of a stranger is worse than a slow run.

One limitation I am documenting rather than hiding: the API runs a single worker. The background
runner keeps run state in process memory, so a second worker would answer "run not found" for
roughly half the progress requests. Scaling out means moving runs onto a real job queue first, and
setting `--workers 4` without doing that would give me a system that looks scalable and is broken.

## What is in the repo

```
src/hirelens/
  ingest/      PDF, DOCX and TXT readers that preserve character offsets and layout
  extract/     PII detection, blind mode, span locator, citation-verified extraction
  retrieve/    evidence chunking, local embeddings, BM25 + dense hybrid retrieval
  assess/      rubric compiler, requirement judge, scoring, risk flags, interview questions
  evals/       golden set, labelling workflow, metrics, baselines, regression gate
  audit/       counterfactual perturbations, fairness runner, bias report
  llm/         async client, response cache, rate limiter, retries, schema repair
  schemas/     Cited[T], evidence primitives, resume and job models
  api/         FastAPI routes, async SQLAlchemy models, background runner, SSE
web/           React dashboard: typed client, live progress, evidence highlighting
tests/         397 Python tests across 11 files, all against a fake provider
scripts/       demo set renderer, eval smoke test, audit smoke test
samples/       synthetic resumes and job descriptions for trying it out
docs/          architecture diagram and screenshots
```

The dashboard is React and React DOM on top of a Vite and TypeScript toolchain, with no UI
framework at all. Its API client is typed by hand against the response models rather than
generated, which makes `npm run typecheck` a real
check: rename a field in `hirelens.api.schemas` without updating the frontend and CI fails. A
generator would have produced a looser surface where the same mistake becomes a runtime
`undefined`.

I also used plain `httpx` instead of three vendor SDKs. Three SDKs means three auth conventions,
three retry behaviours and three async styles to reconcile, and each provider needs about twenty
lines of request shaping.

## Tests

```bash
make check      # ruff + mypy strict + pytest
make test       # Python unit tests, no network, no API key
make web-test   # dashboard typecheck and unit tests
```

397 Python tests and 9 dashboard tests. Everything runs against a fake provider and the
dependency-free embedder, so the suite needs no key, costs nothing, downloads no model, and runs
in CI on Python 3.10 and 3.12 on every push.

The ones worth reading:

| Test | What it pins down |
|---|---|
| `test_samples_are_not_collapsed_by_the_cache` | The k samples use distinct nonces, so the band is measured rather than always zero |
| `test_hallucinated_quote_is_reported_not_cited` | A quote that is not in the document fails verification instead of being emitted |
| `test_null_control_changes_nothing` | The audit's noise floor really is the same resume scored twice |
| `test_audit_detects_a_deliberately_biased_model` | The audit fires on a model rigged to reward elite institutions |
| `test_a_failing_response_is_evicted` | Invalid responses do not poison the cache forever |
| `test_even_split_resolves_downward` | A 2-2 verdict split rounds against the candidate, not for them |
| `test_unmet_must_have_overrides_a_good_score` | A high score that misses a hard requirement ranks below a lower one that meets everything |

## Stack and license

Python 3.10+, Pydantic v2, FastAPI, async SQLAlchemy, PyMuPDF, sentence-transformers with
`bge-small-en-v1.5` on CPU, plain httpx for all three providers. React 18, TypeScript and Vite on
the front end. Ruff and mypy strict, GitHub Actions for CI.

MIT.
