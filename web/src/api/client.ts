/**
 * The single place that talks to the API.
 *
 * Every request funnels through `request()` so that error handling is uniform.
 * FastAPI returns errors as `{"detail": "..."}`; a fetch wrapper that ignores
 * the body surfaces "500 Internal Server Error" to the user and throws away the
 * one sentence that would have explained it. That sentence is often the whole
 * message, for example "the daily free-tier quota is exhausted", so it is worth
 * the extra parse.
 */

import type {
  AssessmentDetail,
  DocumentOut,
  DocumentText,
  Health,
  Job,
  Run,
  RunCreate,
  Shortlist,
  UploadResponse,
} from "./types";

/**
 * Empty by default: the frontend is served from the same origin as the API in
 * the container image, so relative URLs are correct there and the Vite dev
 * server proxies them in development. Only a split deployment needs this set.
 */
const BASE = (import.meta.env["VITE_API_BASE"] as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True for the failures a user can act on by waiting rather than editing. */
  get isQuota(): boolean {
    return this.status === 429 || /quota|rate limit/i.test(this.message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch (cause) {
    // A network-level failure. In practice this nearly always means the API is
    // not running, so say that rather than "Failed to fetch".
    throw new ApiError(
      "Could not reach the API. Is it running on port 8000?",
      0,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI validation errors arrive as a list of location/message pairs.
        detail = body.detail
          .map((item) => {
            const e = item as { loc?: unknown[]; msg?: string };
            const where = Array.isArray(e.loc) ? e.loc.slice(1).join(".") : "";
            return where ? `${where}: ${e.msg}` : (e.msg ?? "invalid");
          })
          .join("; ");
      }
    } catch {
      // Non-JSON error body. Keep the status line.
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),

  listJobs: () => request<Job[]>("/api/jobs"),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  createJob: (description: string, title: string) =>
    request<Job>("/api/jobs", json({ description, title })),
  jobRuns: (id: string) => request<Run[]>(`/api/jobs/${id}/runs`),

  listDocuments: () => request<DocumentOut[]>("/api/documents"),
  documentText: (id: string) =>
    request<DocumentText>(`/api/documents/${id}/text`),

  uploadDocuments: (files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<UploadResponse>("/api/documents", {
      method: "POST",
      body: form,
      // No Content-Type header: the browser must set the multipart boundary.
    });
  },

  createRun: (payload: RunCreate) => request<Run>("/api/runs", json(payload)),
  getRun: (id: string) => request<Run>(`/api/runs/${id}`),
  shortlist: (id: string) => request<Shortlist>(`/api/runs/${id}/shortlist`),

  assessment: (id: string) =>
    request<AssessmentDetail>(`/api/assessments/${id}`),

  /** URL of the original file, for the "open the PDF" link. */
  rawDocumentUrl: (id: string) => `${BASE}/api/documents/${id}/raw`,

  /** URL of the progress stream. Consumed by `useRunProgress`. */
  eventsUrl: (runId: string) => `${BASE}/api/runs/${runId}/events`,
};
