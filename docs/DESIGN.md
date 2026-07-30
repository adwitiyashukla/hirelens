# HireLens: Design Brief

> Evidence-grounded candidate screening. Every score cites its source, every score is
> measured against human judgement, and every score is audited for demographic bias.

**Status:** design locked, implementation in progress
**Author:** Adwitiya Shukla
**Target reader:** you, and any engineer or recruiter who opens the repo

---

## 1. Why most resume-screener projects are weak

Search GitHub for "AI resume screener" and you get several hundred repositories that all do the
same thing:

```
resume.pdf  ->  extract text  ->  "Hey GPT, score this out of 100"  ->  87
```

An experienced engineer looks at that and sees three problems immediately.

**It cannot justify itself.** The number 87 is unattributable. You cannot point at a line of the
resume and say "this is why." If a candidate is rejected and asks why, you have nothing.

**It is not reproducible.** Run it twice and you get 87, then 81. Nobody measures this, so nobody
knows the system is unstable. A screening tool whose output moves 6 points between identical runs
is not a tool, it is a random number generator with a vocabulary.

**It has never been evaluated.** There is no ground truth, no metric, no test that says "this
system agrees with expert human screeners X% of the time." The author has no idea whether it
works. Neither does anyone else.

There is a fourth problem that nobody in that search result even acknowledges: **automated
employment screening is legally regulated.** NYC Local Law 144 requires an annual independent bias
audit of any automated employment decision tool. The EU AI Act classifies employment screening as
high-risk and mandates documentation, logging, and human oversight. A resume scorer that cannot
produce a bias report is not deployable anywhere that matters.

Those four gaps are the whole opportunity. They are also, conveniently, exactly the skills that
distinguish an AI/ML engineer from someone who has read the OpenAI quickstart.

---

## 2. The thesis

> A screening decision is only useful if you can defend it.

HireLens is built around that one sentence. Four properties follow from it, and each one is a
concrete engineering subsystem rather than a slogan.

| Property | What it means | How it is built |
|---|---|---|
| **Grounded** | Every claim the system makes points to an exact character range in the source document | Span-tracked ingestion, citation-carrying schemas, retrieval-scoped judging |
| **Stable** | Re-running the same input produces the same decision, and we publish the variance | Self-consistency sampling, median aggregation, published confidence intervals |
| **Measured** | We know how well it agrees with human experts, in numbers | Golden dataset, Spearman rank correlation, MAE, regression tests in CI |
| **Fair** | We actively test whether demographics move the score, and fail the build if they do | Counterfactual perturbation harness, automated bias report |

If you only remember one thing about this project, remember this: **the model is the easy part.
The evaluation harness is the project.**

---

## 3. What it actually does

The user is a recruiter or hiring manager preparing for a first-round interview.

1. **Paste a job description.** The system compiles it into a structured rubric: a weighted set of
   atomic, checkable requirements. No hardcoded rubric, so it works for a backend role, a design
   role, or a research role without a code change.
2. **Upload a batch of resumes.** PDF, DOCX, or text. Scanned documents fall back to OCR.
3. **Get a ranked shortlist.** Each candidate has an overall fit score with a confidence band and
   a per-requirement breakdown.
4. **Drill into any candidate.** The resume is rendered in the browser with the exact supporting
   sentences highlighted. Hover a requirement, the evidence lights up. Click through to the
   source. This is the demo moment that sells the project.
5. **Get an interview prep pack.** Auto-generated questions targeted at the specific gaps and the
   specific unverified claims in that candidate's resume, not generic questions. This is the
   "so HR has an idea about the candidate before the interview" part of the brief.
6. **Read the risk flags.** Unexplained timeline gaps, claims with no supporting evidence,
   projects with no link, vague impact statements with no numbers.
7. **Open the fairness report.** For the batch just processed: how much did the score move when we
   swapped names, universities, and locations? Published, not hidden.

Blind mode is on by default: names, photos, addresses, and other PII are detected and stripped
before anything reaches the model.

---

## 4. Architecture

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
   github user --> ENRICH (async, cached, GraphQL) ---------------->    |  ASSESSMENT          |
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

### Stage notes

**Ingest.** PyMuPDF gives us text plus bounding boxes. We keep a character-offset map from every
extracted token back to (page, rect) so the frontend can draw a highlight box over the original
PDF. This offset map is what makes citation possible, and it is why we cannot use a naive
`page.get_text()` dump.

**Redact.** A rules-plus-NER pass finds names, emails, phones, addresses, and university names. We
store the spans rather than destroying the text, so blind mode is a toggle rather than a one-way
door, and so the fairness harness can perturb exactly those spans.

**Extract.** Six focused extraction calls (basics, work, education, skills, projects, awards)
rather than one giant prompt. Each returns a Pydantic model where every field is wrapped in
`Cited[T]`, carrying both the value and the span it came from. If the model returns a span that
does not actually contain the claimed text, we reject and repair. This is the single most
important design decision in the project: **the schema makes ungrounded output a validation error
rather than a silent lie.**

**Index.** The resume is split into evidence units (one bullet, one role, one project) and
embedded with `bge-small-en-v1.5` via sentence-transformers. This runs locally on CPU in a couple
of seconds and costs nothing, forever.

**Match.** For each rubric requirement, retrieve the top-k evidence units using hybrid search:
BM25 for exact terminology (a JD asking for "Kubernetes" should hit the resume token "Kubernetes")
fused with dense embeddings for semantic matches ("shipped to prod" matches "production deployment
experience"). Reciprocal rank fusion combines the two rankings.

**Judge.** Here is the key move: the LLM never sees the whole resume when scoring. It sees *one
requirement* and *the handful of evidence units retrieved for it*. This makes each call small,
cheap, parallelisable, and hard to hallucinate in, because there is simply not enough context in
the prompt to invent an unrelated achievement. It also means a bad judgement on one requirement
does not contaminate the others.

**Aggregate.** Each requirement is judged `k=5` times at a non-zero temperature. We take the median
as the score and the interquartile range as the confidence band. High spread on a requirement is
itself a signal: it means the evidence is genuinely ambiguous, and we surface that to the recruiter
as "needs human review" rather than pretending to be certain.

---

## 5. The ML content, in plain terms

You have not built an LLM project before, so here is what each technique is and why a recruiter
cares.

**Embeddings and vector search.** An embedding turns a sentence into a list of roughly 384 numbers,
positioned so that sentences with similar meaning land near each other. We use these to find "the
part of the resume that is about distributed systems" without the words "distributed systems"
needing to appear. Recruiters care because this is the foundation of every RAG system shipping
today.

**Hybrid retrieval and reciprocal rank fusion.** Pure embedding search is bad at exact terms (model
numbers, library names, acronyms). Pure keyword search is bad at paraphrase. Running both and
fusing the rankings beats either alone. Knowing *why* is a mid-level RAG engineering signal.

**Structured generation with schema repair.** We hand the model a JSON schema and validate the
response against it. When validation fails, we feed the validation error back and retry rather than
crashing. Production LLM systems live or die on this loop.

**Grounding and citation verification.** We do not trust the model to tell the truth about where a
claim came from. We verify the span. This is the difference between a demo and a system.

**Self-consistency and uncertainty quantification.** Sampling the same judgement multiple times and
taking the median is a well-established technique for improving LLM reliability. Reporting the
spread as a confidence interval is what turns a point estimate into an honest one.

**Counterfactual fairness testing.** Take a resume, change only the name from one that reads as
male to one that reads as female, hold everything else constant, re-score. Repeat across a matrix
of name, gender-coded pronouns, university tier, and location. If the score moves more than a
threshold, the system is biased along that axis and CI fails the build. This is a real research
technique and almost nobody applies it in a portfolio project.

**LLM-as-judge evaluation.** We build a golden set of resumes with human-assigned rankings, then
measure Spearman rank correlation between our system's ordering and the human ordering. That single
number is the honest answer to "does this thing work."

---

## 6. The evaluation harness

This is the part that makes the project senior. It lives in `evals/` and runs with `make eval`.

**Golden dataset.** 40 to 60 synthetic resumes, generated to span a deliberate quality range,
paired with 3 job descriptions. Synthetic because using real resumes is a privacy problem and
because we need controlled variation. Each resume-JD pair gets a human rank (yours, done carefully
and documented) and the rationale recorded.

**Metrics we report in the README:**

| Metric | What it answers | Target |
|---|---|---|
| Spearman rho vs human ranking | Do we order candidates the way a human would? | > 0.75 |
| MAE on requirement scores | How far off are individual judgements? | < 12 pts |
| Self-consistency sigma | How much does the score wobble across identical runs? | < 4 pts |
| Citation validity rate | What fraction of cited spans actually contain the claim? | > 0.95 |
| Max demographic drift | Worst-case score change from a counterfactual swap | < 2 pts |
| p95 latency per resume | Is it usable? | < 25 s |

**Model comparison.** The same harness runs against `gemini-2.5-flash`, a Groq-hosted Llama model,
and a local `qwen3:4b`. Publishing a table of quality versus cost versus latency across three
models is exactly the kind of empirical work an ML engineer does, and it demonstrates that the
provider abstraction is real rather than decorative.

**Prompt regression tests.** Every prompt change reruns the harness in CI. If rho drops or drift
rises, the PR fails. This turns prompt engineering from vibes into engineering.

---

## 7. Stack

Everything below is free.

| Layer | Choice | Cost |
|---|---|---|
| LLM (primary) | Google Gemini Flash, free tier | Free, generous limits |
| LLM (fallback) | Groq, free tier | Free, very fast |
| LLM (offline) | Ollama, qwen3:4b or gemma3:4b | Free, runs on your machine |
| Embeddings | `bge-small-en-v1.5` via sentence-transformers | Free, local CPU, no API |
| Vector store | Qdrant (sqlite-vec as a simpler fallback) | Free |
| Keyword search | `rank_bm25` | Free, in-process |
| API | FastAPI, async, SSE for live progress | Free |
| Database | Postgres | Free |
| Frontend | React + TypeScript + Vite + Tailwind + TanStack Query | Free |
| PDF rendering | `react-pdf` with span highlight overlay | Free |
| Container | Docker Compose | Free |
| CI | GitHub Actions | Free for public repos |
| Deploy | Fly.io or Render (API), Vercel (frontend) | Free tier |

The embeddings running locally is the important one. It means the expensive, high-volume part of
the pipeline never touches a paid API, and anyone who clones the repo can run it without a key.

---

## 8. Build plan

Each phase ends with something that runs and is committed. No phase depends on a later one.

**Phase 1: Foundation.** Repo scaffold, settings, evidence-span schemas, offset-preserving
ingestion for PDF/DOCX, async provider layer over Gemini/Groq/Ollama with retries, rate limits, and
an on-disk response cache. Ends with: `hirelens ingest resume.pdf` prints structured text with
offsets.

**Phase 2: Extraction.** Per-section extraction with `Cited[T]` schemas, span verification, schema
repair loop, PII detection. Ends with: `hirelens parse resume.pdf` emits a validated JSON resume
where every field carries a verified citation.

**Phase 3: Rubric and retrieval.** JD compiler, evidence chunking, local embeddings, vector store,
BM25, reciprocal rank fusion. Ends with: `hirelens match resume.pdf jd.txt` shows which resume
lines answer which requirement.

**Phase 4: Assessment.** Per-requirement judging, self-consistency sampling, weighted aggregation,
risk flags, interview question generation. Ends with: `hirelens score` produces a full assessment
with confidence bands.

**Phase 5: Evaluation harness.** Golden set generation, human labelling, metrics, model comparison
table, CI regression gate. Ends with: `make eval` prints the metrics table.

**Phase 6: Fairness audit.** Counterfactual perturbation matrix, drift measurement, report
generation, CI gate. Ends with: `make audit` emits a bias report.

**Phase 7: API and workers.** FastAPI, Postgres, background jobs, SSE progress streaming, batch
upload, OpenAPI docs.

**Phase 8: Frontend.** Upload, JD editor, ranked table, candidate detail with evidence
highlighting, live run progress.

Built as planned with two deliberate departures.

*Highlighting is over the extracted text, not an overlay on a rendered PDF.* The API already
returns highlight rectangles, so the overlay is a supported future addition rather than a
missing capability. But `react-pdf` and its worker bundle roughly triple the dependency
footprint, and the overlay is only correct for PDFs: a DOCX or TXT resume has no page geometry
to draw on. Highlighting the canonical text works for every supported format, is the same
coordinate system the citations are stored in, and needs no dependency at all. The rectangles
stay in the API for whoever wants the overlay.

*No fairness or eval dashboard page.* Both of those artefacts are reports produced by CLI runs
that need an API key and, for the eval set, human labels. A page rendering numbers that do not
exist yet would be a mockup. Instead there is a "How it works" page stating each of the four
claims, the mechanism enforcing it, and the exact command that reproduces it. When the reports
exist, they are markdown in `docs/`, which is where a reader expects an audit to live.

The dependency list is React plus three dev tools. No UI framework, no state library, no router,
no data-fetching library. Each was considered and each would have cost more than it returned at
this size, which is a judgement worth being able to defend in an interview either way.

**Phase 9: Ship.** Docker Compose, GitHub Actions, deployment configuration, architecture
diagram, README.

The image is multi-stage: Node builds the dashboard, Python serves it alongside the API from one
process, so a free hosting tier that grants one container is enough. `render.yaml` and
`fly.toml` cover the two realistic free options. CI gained a job that typechecks, tests and
builds the frontend, which matters because the API client is hand-typed against the response
models and therefore actually catches a schema drift.

Not done, and deliberately: the demo GIF and the screenshots. Both require a live run, and the
free-tier daily quota was exhausted during development. `docs/screenshots/README.md` specifies
exactly which five captures to take and what each has to show. Committing a mocked screenshot to
a project whose thesis is that unevidenced claims are worthless would be self-refuting.

---

## 9. How to present it

The README is the product as far as a recruiter is concerned. It should lead with, in order:

1. One sentence on what it does and the defensibility thesis.
2. A demo GIF of the evidence-highlight interaction. This is the thing people remember.
3. The metrics table from `make eval`. Real numbers, honestly reported, including the ones that are
   not flattering.
4. The architecture diagram.
5. The fairness report summary.
6. A "what I learned and what I would do differently" section. Nothing signals seniority faster
   than accurately describing your own system's limitations.

Do not write "powered by cutting-edge AI." Write "Spearman rho = 0.78 against human ranking on a
54-resume golden set, max demographic drift 1.3 points." The second one gets you the interview.

---

## 10. Open questions to resolve during the build

- Golden set labelling is the bottleneck and the part only you can do. Budget real time for it.
- Qdrant may be overkill at this scale. If it complicates the demo deploy, `sqlite-vec` is a
  legitimate simplification and worth documenting as a deliberate trade-off.
- OCR fallback (Tesseract) is genuinely fiddly. It is a Phase 2 stretch goal, not a blocker.
- The interview question generator is the feature most likely to impress non-technical viewers and
  least likely to impress engineers. Build it, but do not lead with it.
