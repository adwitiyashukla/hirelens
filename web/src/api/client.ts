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

const BASE = (import.meta.env["VITE_API_BASE"] as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get isQuota(): boolean {
    return this.status === 429 || /quota|rate limit/i.test(this.message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch (cause) {

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

        detail = body.detail
          .map((item) => {
            const e = item as { loc?: unknown[]; msg?: string };
            const where = Array.isArray(e.loc) ? e.loc.slice(1).join(".") : "";
            return where ? `${where}: ${e.msg}` : (e.msg ?? "invalid");
          })
          .join("; ");
      }
    } catch {

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

    });
  },

  createRun: (payload: RunCreate) => request<Run>("/api/runs", json(payload)),
  getRun: (id: string) => request<Run>(`/api/runs/${id}`),
  shortlist: (id: string) => request<Shortlist>(`/api/runs/${id}/shortlist`),

  assessment: (id: string) =>
    request<AssessmentDetail>(`/api/assessments/${id}`),

  rawDocumentUrl: (id: string) => `${BASE}/api/documents/${id}/raw`,

  eventsUrl: (runId: string) => `${BASE}/api/runs/${runId}/events`,
};
