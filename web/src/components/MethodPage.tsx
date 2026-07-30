/**
 * What the system claims, and how each claim is checked.
 *
 * This page exists because the interesting part of the project is not that an
 * LLM can read a resume. It is the four properties below, each of which is
 * enforced by a subsystem rather than asserted in a README. The commands are
 * included so a reader can reproduce the numbers rather than take them on
 * trust.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Health } from "../api/types";

interface Claim {
  title: string;
  problem: string;
  mechanism: string;
  check: string;
  command: string;
}

const CLAIMS: Claim[] = [
  {
    title: "Grounded",
    problem:
      "A model asked to summarise a resume will confidently describe experience the candidate does not have. The output reads exactly as well when it is wrong.",
    mechanism:
      "Every scored claim is typed as Cited[T]: a value plus character spans into the source document. The model returns verbatim quotes, never offsets, because models cannot count characters. A real string search locates each quote, falling back from exact match to whitespace-normalised match to a fuzzy sliding window.",
    check:
      "A quote that cannot be located is a validation error, not a warning. The assessment is rejected before it is stored, so an ungrounded score cannot reach the shortlist.",
    command: "pytest tests/test_extract.py -k grounding",
  },
  {
    title: "Stable",
    problem:
      "Ask the same model the same question twice and the answers differ. A single sample presented as a score implies a precision that does not exist.",
    mechanism:
      "Each requirement is judged k times at non-zero temperature, with a per-sample nonce so the cache cannot collapse them into one answer. Verdicts are ordinal, and the aggregate is the median rather than the mean, so one outlier sample cannot drag a verdict. Ties resolve downward.",
    check:
      "Scores are published with a bootstrap percentile interval and a per-requirement agreement rate. Requirements whose samples disagreed are flagged in the detail view instead of being silently averaged.",
    command: "hirelens score resume.pdf --jd role.txt",
  },
  {
    title: "Measured",
    problem:
      "Almost every resume-screening project reports zero accuracy numbers, because measuring requires labels and labels require work.",
    mechanism:
      "A deterministic generator builds a golden set of synthetic resumes with known ground truth, labelled through a CLI. Results are scored against three baselines: random ordering, keyword overlap, and embedding similarity. Rank correlation uses tie-corrected Spearman and Kendall tau-b.",
    check:
      "The regression gate fails CI if quality drops below the recorded baseline, so a prompt change that quietly makes things worse cannot merge.",
    command: "make golden && make label && make eval",
  },
  {
    title: "Fair",
    problem:
      "A screening tool can be biased in ways no accuracy metric detects, because the bias is consistent and therefore invisible to a correlation.",
    mechanism:
      "A counterfactual audit in the design of Bertrand and Mullainathan: hold the resume fixed, vary only a demographic proxy (gender-coded name, ethnicity-coded name, university prestige, location), and measure the score drift. A null control perturbs nothing and establishes the noise floor.",
    check:
      "Drift is only reportable if it exceeds the null control. The audit gate fails the build when demographic drift exceeds the configured threshold.",
    command: "python -m hirelens.audit.cli run --budget tiny --gate",
  },
];

export function MethodPage() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
  }, []);

  return (
    <div className="page">
      <div className="card">
        <header>
          <h1>How this works</h1>
          {health && (
            <span className="row">
              <span className="tag">{health.provider}</span>
              <span className="tag mono">{health.model}</span>
              <span className={`badge ${health.provider_configured ? "strong" : "danger"}`}>
                {health.provider_configured ? "provider ready" : "no credential"}
              </span>
            </span>
          )}
        </header>
        <p className="hint" style={{ maxWidth: 760 }}>
          Reading a resume with a language model is the easy part and takes about
          forty lines. The hard part is that the output of those forty lines is
          not trustworthy enough to put in front of a hiring decision. Four
          properties are what make the difference, and each one is enforced by
          code rather than claimed in prose.
        </p>
      </div>

      {CLAIMS.map((claim) => (
        <div className="card" key={claim.title}>
          <header>
            <h2>{claim.title}</h2>
          </header>
          <div className="claim">
            <div>
              <div className="label">The failure it prevents</div>
              <p>{claim.problem}</p>
            </div>
            <div>
              <div className="label">Mechanism</div>
              <p>{claim.mechanism}</p>
            </div>
            <div>
              <div className="label">How it is enforced</div>
              <p>{claim.check}</p>
            </div>
            <code className="command">{claim.command}</code>
          </div>
        </div>
      ))}

      <div className="card">
        <header>
          <h2>What this deliberately does not do</h2>
        </header>
        <ul style={{ margin: 0, paddingLeft: 18, color: "var(--muted)", maxWidth: 800 }}>
          <li>
            It does not reject candidates. It produces a reviewed shortlist with
            the evidence attached, and a human makes the decision.
          </li>
          <li>
            It does not infer protected characteristics, and the rubric compiler
            explicitly refuses to turn demographic proxies into requirements.
          </li>
          <li>
            It does not rank on a score alone. A candidate who misses a hard
            requirement sits below one who meets them all, whatever the totals.
          </li>
          <li>
            It does not claim a fairness guarantee. It claims a measurement,
            taken with a null control, that anyone can rerun.
          </li>
        </ul>
      </div>
    </div>
  );
}
