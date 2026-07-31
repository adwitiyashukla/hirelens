/**
 * The screening flow: describe the job, add resumes, run, watch, rank.
 *
 * Kept on one page deliberately. A wizard would hide the job description while
 * the results are read, and the first question anyone asks about a low score is
 * "what did it think the job needed?". The compiled rubric stays visible.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  DocumentOut,
  Job,
  RejectedUpload,
  Run,
  ShortlistEntry,
} from "../api/types";
import { useRunProgress } from "../hooks/useRunProgress";
import { BandBadge, RateBadge, ScoreCell } from "./Badges";
import { Alert, Check, initialsOf, Upload } from "./Icons";

const SAMPLE_JD = `Senior Backend Engineer

We are looking for a backend engineer to own the services behind our payments
platform. You will design APIs, tune the data layer, and mentor two junior
engineers.

Requirements
- 5+ years building production backend services
- Strong Python, including async
- Deep PostgreSQL experience: schema design, query tuning, migrations
- Experience running services on Kubernetes
- Track record of mentoring engineers

Nice to have
- Payments or fintech domain experience
- Go or Rust
- Open source contributions`;

export function ScreenPage({
  onOpenCandidate,
}: {
  onOpenCandidate: (assessmentId: string) => void;
}) {
  const [description, setDescription] = useState("");
  const [title, setTitle] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [compiling, setCompiling] = useState(false);

  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rejected, setRejected] = useState<RejectedUpload[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const [blindMode, setBlindMode] = useState(true);
  const [run, setRun] = useState<Run | null>(null);
  const [entries, setEntries] = useState<ShortlistEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fileInput = useRef<HTMLInputElement>(null);
  const { progress, finished } = useRunProgress(run?.id ?? null);

  useEffect(() => {
    api.listDocuments().then(setDocuments).catch(() => undefined);
  }, []);

  // The stream reports completion; the ranked table then comes from the
  // database. Deriving the shortlist from the events instead would mean
  // reimplementing the ranking rule in the browser, and the two would drift.
  useEffect(() => {
    if (!finished || !run) return;
    api
      .shortlist(run.id)
      .then((shortlist) => {
        setEntries(shortlist.entries);
        setRun(shortlist.run);
      })
      .catch((cause) =>
        setError(cause instanceof ApiError ? cause.message : String(cause)),
      );

    // The rubric is compiled during the run, so this is the first moment it
    // exists. Re-fetching here is what makes it appear beside the results,
    // which is also where a reader wants it: the first question a low score
    // prompts is "what did it think the job required?".
    if (job) {
      api
        .getJob(job.id)
        .then(setJob)
        .catch(() => undefined);
    }
  }, [finished, run?.id]);

  const saveJob = async () => {
    setError(null);
    setCompiling(true);
    try {
      const created = await api.createJob(description, title);
      setJob(created);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setCompiling(false);
    }
  };

  const upload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      const response = await api.uploadDocuments(files);
      setRejected(response.rejected);
      setDocuments(await api.listDocuments());
      // Auto-select what was just uploaded, including duplicates: a recruiter
      // who re-uploads a resume still means to screen that candidate.
      setSelected((current) => {
        const next = new Set(current);
        for (const item of response.uploaded) next.add(item.document.id);
        return next;
      });
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setUploading(false);
    }
  }, []);

  const start = async () => {
    if (!job) return;
    setError(null);
    setEntries([]);
    try {
      const created = await api.createRun({
        job_id: job.id,
        document_ids: [...selected],
        blind_mode: blindMode,
        top_k: 4,
        with_questions: true,
      });
      setRun(created);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const toggle = (id: string) =>
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const done = (progress?.completed ?? 0) + (progress?.failed ?? 0);
  const total = progress?.total ?? 0;
  const running = run !== null && !finished;

  return (
    <div className="page">
      {error && (
        <div className="banner error">
          <Alert />
          <span>{error}</span>
        </div>
      )}

      <div className="columns">
        <div>
          <div className="card">
            <header>
              <h2 className={`step ${job ? "done" : ""}`}>
                <span className="num">{job ? <Check /> : "1"}</span>
                Job description
              </h2>
              <button
                className="ghost"
                onClick={() => {
                  setDescription(SAMPLE_JD);
                  setTitle("Senior Backend Engineer");
                }}
              >
                Use sample
              </button>
            </header>
            <p className="hint">
              The description is compiled into weighted, atomic requirements
              before any resume is read, and every candidate in the run is
              scored against that one fixed rubric. Compiling per candidate
              instead would give each a slightly different rubric and make the
              scores incomparable. The compiled rubric appears here once the
              first run has produced it.
            </p>

            <label className="field">
              <span>Title</span>
              <input
                type="text"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Senior Backend Engineer"
              />
            </label>
            <label className="field">
              <span>Full posting</span>
              <textarea
                rows={12}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Paste the complete job posting here"
              />
            </label>

            <div className="row">
              <button
                className="primary"
                disabled={description.trim().length < 80 || compiling}
                onClick={saveJob}
              >
                {compiling ? (
                  <>
                    <span className="spin" /> Saving
                  </>
                ) : (
                  "Save job description"
                )}
              </button>
              {job && !job.requirements.length && (
                <span className="badge neutral">saved</span>
              )}
              {description.trim().length > 0 && description.trim().length < 80 && (
                <span className="hint" style={{ margin: 0 }}>
                  Needs at least 80 characters to compile a useful rubric
                </span>
              )}
            </div>
          </div>

          {/*
            Rendered only once requirements exist. The API compiles the rubric
            lazily on the first run, so between saving the job and finishing a
            run there is genuinely nothing to show, and an empty table with a
            "0 points" badge reads like a failure rather than a pending step.
          */}
          {job && job.requirements.length > 0 && (
            <div className="card">
              <header>
                <h2>Compiled rubric</h2>
                <span className="tag num">
                  {job.requirements.reduce((sum, r) => sum + r.weight, 0).toFixed(0)} points
                </span>
              </header>
              <table className="grid">
                <thead>
                  <tr>
                    <th>Requirement</th>
                    <th>Kind</th>
                    <th style={{ textAlign: "right" }}>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {job.requirements.map((requirement) => (
                    <tr key={requirement.requirement_id} style={{ cursor: "default" }}>
                      <td>{requirement.text}</td>
                      <td>
                        <span
                          className={`badge ${
                            requirement.kind === "must_have" ? "partial" : "neutral"
                          }`}
                        >
                          {requirement.kind === "must_have" ? "must have" : "nice to have"}
                        </span>
                      </td>
                      <td className="num" style={{ textAlign: "right" }}>
                        {requirement.weight.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <header>
              <h2 className={`step ${selected.size > 0 ? "done" : ""}`}>
                <span className="num">{selected.size > 0 ? <Check /> : "2"}</span>
                Resumes
              </h2>
              <span className="tag">{selected.size} selected</span>
            </header>

            <div
              className={`dropzone ${dragging ? "over" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                void upload([...event.dataTransfer.files]);
              }}
            >
              {uploading ? (
                <>
                  <span className="spin" /> Extracting text and building the offset map
                </>
              ) : (
                <>
                  <Upload className="icon" />
                  <strong>Drop resumes here</strong>, or click to choose
                  <div style={{ fontSize: 11.5, color: "var(--text-faint)", marginTop: 4 }}>
                    PDF, DOCX or TXT. Uploading the same file twice reuses the
                    first one rather than creating a duplicate candidate.
                  </div>
                </>
              )}
              <input
                ref={fileInput}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                style={{ display: "none" }}
                onChange={(event) => {
                  void upload([...(event.target.files ?? [])]);
                  event.target.value = "";
                }}
              />
            </div>

            {rejected.length > 0 && (
              <div className="banner warn" style={{ marginTop: 12 }}>
                Could not read {rejected.length} file
                {rejected.length > 1 ? "s" : ""}:{" "}
                {rejected.map((r) => `${r.filename} (${r.reason})`).join("; ")}
              </div>
            )}

            {documents.length > 0 && (
              <ul className="filelist">
                {documents.map((document) => (
                  <li key={document.id}>
                    <label className="checkline" style={{ margin: 0 }}>
                      <input
                        type="checkbox"
                        checked={selected.has(document.id)}
                        onChange={() => toggle(document.id)}
                      />
                      {document.filename}
                    </label>
                    <span className="tag num">
                      {document.page_count}p / {document.char_count.toLocaleString()} chars
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="card">
            <header>
              <h2 className={`step ${entries.length > 0 ? "done" : ""}`}>
                <span className="num">{entries.length > 0 ? <Check /> : "3"}</span>
                Screen
              </h2>
            </header>
            <label className="checkline">
              <input
                type="checkbox"
                checked={blindMode}
                onChange={(event) => setBlindMode(event.target.checked)}
              />
              Blind mode: redact names, addresses, and contact details before the
              model sees the resume
            </label>
            <p className="hint">
              Redaction is length-preserving, so character offsets stay valid and
              citations still point at the right lines in the original file.
            </p>
            <button
              className="primary"
              disabled={!job || selected.size === 0 || running}
              onClick={start}
            >
              {running ? (
                <>
                  <span className="spin" /> Screening
                </>
              ) : (
                `Screen ${selected.size || ""} candidate${selected.size === 1 ? "" : "s"}`
              )}
            </button>

            {progress && (
              <>
                <div className={`progress ${running ? "live" : ""}`}>
                  <div style={{ width: `${total ? (done / total) * 100 : 0}%` }} />
                </div>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="hint" style={{ margin: 0 }}>
                    {progress.stage}
                    {progress.message ? `: ${progress.message}` : ""}
                  </span>
                  <span className="tag num">
                    {done} / {total}
                  </span>
                </div>
                {progress.failed > 0 && (
                  <div className="banner warn" style={{ marginTop: 8 }}>
                    {progress.failed} candidate{progress.failed > 1 ? "s" : ""} could
                    not be assessed. Partial results are still shown below, and the
                    failures are excluded rather than scored as zero.
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {entries.length > 0 && (
        <div className="card">
          <header>
            <h2>Shortlist</h2>
            <span className="hint" style={{ margin: 0 }}>
              Ranked by must-have compliance first, then score
            </span>
          </header>

          {/*
            The aggregate quality numbers sit above the ranking on purpose. They
            answer "should I trust this table at all?", which has to be settled
            before the ordering inside it means anything.
          */}
          <div className="metrics" style={{ marginBottom: 18 }}>
            <div className="metric">
              <div className="label">Screened</div>
              <div className="value">{entries.length}</div>
              <div className="note">
                {entries.filter((e) => e.meets_must_haves).length} meet every must-have
              </div>
            </div>
            <div className="metric">
              <div className="label">Grounded</div>
              <div className="value">
                {(
                  (entries.reduce((sum, e) => sum + e.grounding_rate, 0) /
                    entries.length) *
                  100
                ).toFixed(0)}
                %
              </div>
              <div className="note">claims carrying a citation</div>
            </div>
            <div className="metric">
              <div className="label">Citations valid</div>
              <div className="value">
                {(
                  (entries.reduce((sum, e) => sum + e.citation_validity_rate, 0) /
                    entries.length) *
                  100
                ).toFixed(0)}
                %
              </div>
              <div className="note">re-checked against the source</div>
            </div>
            <div className="metric">
              <div className="label">Agreement</div>
              <div className="value">
                {(
                  (entries.reduce((sum, e) => sum + e.mean_agreement, 0) /
                    entries.length) *
                  100
                ).toFixed(0)}
                %
              </div>
              <div className="note">across repeated samples</div>
            </div>
          </div>
          <table className="grid">
            <thead>
              <tr>
                <th style={{ width: 34 }}>#</th>
                <th>Candidate</th>
                <th style={{ width: 190 }}>Score</th>
                <th>Band</th>
                <th>Must-haves</th>
                <th title="Share of judged claims that carry a verified citation">
                  Grounded
                </th>
                <th title="Agreement across repeated samples of the same judgement">
                  Agreement
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry, index) => (
                <tr key={entry.id} onClick={() => onOpenCandidate(entry.id)}>
                  <td>
                    <span className={`rank ${index < 3 ? "top" : ""}`}>{index + 1}</span>
                  </td>
                  <td>
                    <div className="who">
                      <span className="disc">{initialsOf(entry.candidate_label)}</span>
                      <span>
                        <div className="name">{entry.candidate_label}</div>
                        <div className="sub">
                          {entry.elapsed_s.toFixed(1)}s to assess
                        </div>
                      </span>
                    </div>
                  </td>
                  <td>
                    <ScoreCell
                      score={entry.score}
                      low={entry.score_low}
                      high={entry.score_high}
                    />
                  </td>
                  <td>
                    <BandBadge band={entry.band} />
                  </td>
                  <td>
                    {entry.meets_must_haves ? (
                      <span className="badge strong">all met</span>
                    ) : (
                      <span className="badge danger">gap</span>
                    )}
                  </td>
                  <td>
                    <RateBadge value={entry.grounding_rate} />
                  </td>
                  <td>
                    <RateBadge value={entry.mean_agreement} good={0.8} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
            Intervals that overlap mean the ordering between those candidates is
            not statistically meaningful. Treat them as a set to review, not a
            ranking to follow.
          </p>
        </div>
      )}
    </div>
  );
}
