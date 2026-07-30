/**
 * One candidate, opened.
 *
 * Left: the score decomposed into requirements, plus risks and the interview
 * questions. Right: the resume, with the cited evidence highlighted.
 *
 * The two panes share `activeRequirementId` in both directions. Selecting a
 * requirement highlights its evidence; clicking a highlight selects the
 * requirement it belongs to. That round trip is what turns "the model says 72"
 * into "here are the four lines that produced 72, and here is what it could not
 * find".
 */

import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { AssessmentDetail, DocumentText, RiskLevel } from "../api/types";
import { BandBadge, RateBadge, ScoreCell, VerdictBadge } from "./Badges";
import { EvidenceView, spansFrom } from "./EvidenceView";

const RISK_TONE: Record<RiskLevel, string> = {
  info: "neutral",
  warning: "partial",
  high: "danger",
};

export function CandidateDetail({
  assessmentId,
  onBack,
}: {
  assessmentId: string;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<AssessmentDetail | null>(null);
  const [document, setDocument] = useState<DocumentText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setDocument(null);
    setError(null);
    setActiveId(null);

    (async () => {
      try {
        const assessment = await api.assessment(assessmentId);
        if (cancelled) return;
        setDetail(assessment);

        // Fetched second and separately: the resume text is much larger than
        // the assessment, and the requirement list should render immediately
        // rather than waiting on it.
        const text = await api.documentText(assessment.document.id);
        if (!cancelled) setDocument(text);
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : String(cause));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [assessmentId]);

  if (error) {
    return (
      <div className="page">
        <div className="banner error">{error}</div>
        <button className="ghost" onClick={onBack}>
          Back to shortlist
        </button>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="page">
        <div className="empty">
          <span className="spin" /> Loading assessment
        </div>
      </div>
    );
  }

  const mustHaves = detail.requirements.filter((r) => r.kind === "must_have");
  const unmet = mustHaves.filter((r) => r.verdict === "none" || r.verdict === "weak");

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 14 }}>
        <button className="ghost" onClick={onBack}>
          Back to shortlist
        </button>
        <h1 style={{ marginLeft: 4 }}>{detail.candidate_label}</h1>
        <BandBadge band={detail.band} />
        {!detail.meets_must_haves && (
          <span className="badge danger">Misses a must-have</span>
        )}
        <div style={{ flex: 1 }} />
        <a
          className="tag"
          href={api.rawDocumentUrl(detail.document.id)}
          target="_blank"
          rel="noreferrer"
        >
          Open original file
        </a>
      </div>

      <div className="card">
        <div className="metrics">
          <div className="metric">
            <div className="label">Score</div>
            <ScoreCell
              score={detail.score}
              low={detail.score_low}
              high={detail.score_high}
            />
            <div className="note">90% bootstrap interval</div>
          </div>
          <div className="metric">
            <div className="label">Grounding</div>
            <div className="value">{(detail.grounding_rate * 100).toFixed(0)}%</div>
            <div className="note">claims carrying a citation</div>
          </div>
          <div className="metric">
            <div className="label">Citations valid</div>
            <div className="value">
              {(detail.citation_validity_rate * 100).toFixed(0)}%
            </div>
            <div className="note">quotes found in the source</div>
          </div>
          <div className="metric">
            <div className="label">Agreement</div>
            <div className="value">{(detail.mean_agreement * 100).toFixed(0)}%</div>
            <div className="note">across repeated samples</div>
          </div>
        </div>
      </div>

      {unmet.length > 0 && (
        <div className="banner warn">
          Unmet must-have{unmet.length > 1 ? "s" : ""}:{" "}
          {unmet.map((r) => r.requirement_text).join("; ")}
        </div>
      )}

      <div className="columns">
        <div>
          <div className="card">
            <header>
              <h2>Requirements</h2>
              <span className="hint" style={{ margin: 0 }}>
                Select one to locate its evidence
              </span>
            </header>

            {detail.requirements.map((requirement) => {
              const isActive = activeId === requirement.requirement_id;
              return (
                <div
                  key={requirement.requirement_id}
                  className={`req v-${requirement.verdict} ${isActive ? "active" : ""}`}
                  onClick={() =>
                    setActiveId(isActive ? null : requirement.requirement_id)
                  }
                >
                  <header>
                    <div className="text">
                      {requirement.requirement_text}{" "}
                      {requirement.kind === "must_have" && (
                        <span className="tag">must have</span>
                      )}
                    </div>
                    <VerdictBadge verdict={requirement.verdict} />
                  </header>

                  <div className="row" style={{ marginTop: 6, gap: 8 }}>
                    <span className="tag num">
                      {requirement.points.toFixed(1)} / {requirement.max_points.toFixed(1)} pts
                    </span>
                    <span className="tag num">
                      weight {requirement.weight.toFixed(0)}
                    </span>
                    {requirement.is_ambiguous && (
                      <span className="badge partial" title="The repeated samples disagreed enough that this verdict should be treated as uncertain">
                        samples disagreed
                      </span>
                    )}
                    {requirement.citations.length > 0 && (
                      <span className="tag">
                        {requirement.citations.length} citation
                        {requirement.citations.length > 1 ? "s" : ""}
                      </span>
                    )}
                  </div>

                  <div className="reasoning">{requirement.reasoning}</div>

                  {isActive && requirement.citations.length > 0 && (
                    <div className="quotes">
                      {requirement.citations.map((citation, index) => (
                        <blockquote key={index}>
                          "{citation.quote}"
                          {citation.page !== null && (
                            <span className="tag" style={{ marginLeft: 6 }}>
                              p{citation.page}
                            </span>
                          )}
                          {!citation.verified && (
                            <span className="badge danger" style={{ marginLeft: 6 }}>
                              unverified
                            </span>
                          )}
                        </blockquote>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {detail.risks.length > 0 && (
            <div className="card">
              <header>
                <h2>Risk flags</h2>
              </header>
              <p className="hint">
                Flags are shown to the reviewer and never applied to the score. A
                gap in employment is a question to ask, not a penalty to apply.
              </p>
              {detail.risks.map((risk, index) => (
                <div key={index} className="row" style={{ marginBottom: 8 }}>
                  <span className={`badge ${RISK_TONE[risk.level]}`}>{risk.level}</span>
                  <span>{risk.message}</span>
                </div>
              ))}
            </div>
          )}

          {detail.questions.length > 0 && (
            <div className="card">
              <header>
                <h2>Suggested interview questions</h2>
              </header>
              <p className="hint">
                Generated from the weakest and most uncertain requirements, so the
                interview spends its time where the resume was least conclusive.
              </p>
              {detail.questions.map((question, index) => (
                <div key={index} style={{ marginBottom: 12 }}>
                  <div style={{ fontWeight: 550 }}>{question.question}</div>
                  <div className="reasoning">{question.rationale}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card evidence">
          <header>
            <h2>Evidence</h2>
            <div className="row">
              {activeId && (
                <button className="ghost" onClick={() => setActiveId(null)}>
                  Show all
                </button>
              )}
              <RateBadge
                value={detail.citation_validity_rate}
                label="Share of quotes re-verified against the stored document just now"
              />
            </div>
          </header>
          <p className="hint">
            {detail.document.filename} - every highlight is a character span the
            judge cited. Quotes are re-checked against the stored text when this
            page loads, not merely trusted from when they were written.
          </p>

          {document ? (
            <EvidenceView
              text={document.text}
              spans={spansFrom(detail.requirements)}
              activeRequirementId={activeId}
              onSelectRequirement={setActiveId}
            />
          ) : (
            <div className="empty">
              <span className="spin" /> Loading resume text
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
