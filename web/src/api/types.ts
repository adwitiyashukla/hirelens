/**
 * Types mirroring `hirelens.api.schemas`.
 *
 * Hand-written rather than generated from the OpenAPI document. A generator
 * would produce a wider surface (every field optional, every enum a bare
 * string) and the mismatch would then be caught at runtime instead of at
 * compile time. These are narrow on purpose: `Verdict` and `Band` are unions,
 * so a typo in a comparison is a build error.
 */

export type Verdict = "strong" | "clear" | "partial" | "weak" | "none";

/**
 * The exact strings `CandidateAssessment.band` returns.
 *
 * These are already display text, so the frontend renders them rather than
 * translating them. An earlier version of this file declared a different
 * vocabulary invented here, the label lookup missed on every value, and the
 * band column rendered as an empty pill. Typing it against the source is what
 * turns that into a build error instead of a blank cell.
 */
export type Band =
  | "strong fit"
  | "possible fit"
  | "weak fit"
  | "not a fit"
  | "missing a must-have";
export type RunStatus = "queued" | "running" | "completed" | "failed";
export type RiskLevel = "info" | "warning" | "high";
export type RequirementKind = "must_have" | "nice_to_have";

export interface Requirement {
  requirement_id: string;
  text: string;
  kind: RequirementKind;
  category: string;
  weight: number;
  evidence_hint: string;
}

export interface Job {
  id: string;
  title: string;
  description: string;
  rubric_id: string | null;
  created_at: string;
  requirements: Requirement[];
}

export interface DocumentOut {
  id: string;
  filename: string;
  source_format: string;
  page_count: number;
  char_count: number;
  created_at: string;
}

export interface UploadResult {
  document: DocumentOut;
  created: boolean;
}

export interface RejectedUpload {
  filename: string;
  reason: string;
}

export interface UploadResponse {
  uploaded: UploadResult[];
  rejected: RejectedUpload[];
}

export interface Run {
  id: string;
  job_id: string;
  status: RunStatus;
  stage: string;
  total: number;
  completed: number;
  failed: number;
  blind_mode: boolean;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunProgress {
  run_id: string;
  status: RunStatus;
  stage: string;
  total: number;
  completed: number;
  failed: number;
  message: string;
}

export interface ShortlistEntry {
  id: string;
  document_id: string;
  candidate_label: string;
  score: number;
  score_low: number;
  score_high: number;
  band: Band;
  meets_must_haves: boolean;
  mean_agreement: number;
  grounding_rate: number;
  citation_validity_rate: number;
  elapsed_s: number;
}

export interface Shortlist {
  run: Run;
  entries: ShortlistEntry[];
}

export interface HighlightBox {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Citation {
  start: number;
  end: number;
  page: number | null;
  quote: string;
  verified: boolean;
  boxes: HighlightBox[];
}

export interface RequirementResult {
  requirement_id: string;
  requirement_text: string;
  kind: RequirementKind;
  weight: number;
  verdict: Verdict;
  points: number;
  max_points: number;
  agreement: number;
  is_ambiguous: boolean;
  reasoning: string;
  citations: Citation[];
}

export interface Risk {
  code: string;
  level: RiskLevel;
  message: string;
}

export interface Question {
  question: string;
  rationale: string;
  targets: string;
}

export interface AssessmentDetail {
  id: string;
  run_id: string;
  document: DocumentOut;
  candidate_label: string;
  score: number;
  score_low: number;
  score_high: number;
  band: Band;
  meets_must_haves: boolean;
  mean_agreement: number;
  grounding_rate: number;
  citation_validity_rate: number;
  requirements: RequirementResult[];
  risks: Risk[];
  questions: Question[];
}

export interface DocumentText {
  document_id: string;
  filename: string;
  page_count: number;
  text: string;
  blocks: Array<Record<string, unknown>>;
}

export interface Health {
  status: string;
  version: string;
  database: boolean;
  provider: string;
  model: string;
  provider_configured: boolean;
  blind_mode: boolean;
}

export interface RunCreate {
  job_id: string;
  document_ids: string[];
  blind_mode: boolean;
  top_k: number;
  with_questions: boolean;
}
