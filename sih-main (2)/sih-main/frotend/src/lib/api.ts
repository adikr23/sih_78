import type {
  Alert,
  AlertsQuery,
  ApiErrorShape,
  AmphanField,
  AmphanOverview,
  AmphanAlert,
  AmphanTracksResponse,
  HealthResponse,
  Metrics,
  PipelineJobStatus,
  PipelineResult,
  PipelineRunRequest,
  ReplayEvents,
  TiggeAlert,
  TiggeOverview,
  TiggeTracksResponse,
} from "./types";

export const API_BASE_URL = (import.meta.env["VITE_API_BASE_URL"] || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly endpoint: string,
    public readonly developerDetail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const startedAt = performance.now();
  const init: RequestInit = {
    ...options,
    headers: { Accept: "application/json", ...options.headers },
  };
  if (signal) init.signal = signal;
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (import.meta.env["DEV"]) {
    console.debug("[STORMTRACE_API]", {
      path,
      status: response.status,
      elapsed_ms: Number((performance.now() - startedAt).toFixed(1)),
      payload_bytes: Number(response.headers.get("content-length") ?? 0) || undefined,
      cache: response.headers.get("x-cache") ?? response.headers.get("age") ?? "unspecified",
    });
  }
  if (!response.ok) {
    let body: ApiErrorShape = {};
    try { body = (await response.json()) as ApiErrorShape; } catch { /* non-JSON upstream */ }
    const fallback = response.status === 404 ? "Requested record was not found." : "The STORMTRACE engine rejected this request.";
    throw new ApiError(body.detail || fallback, response.status, path, body.detail);
  }
  return response.json() as Promise<T>;
}

const jsonPost = (body?: unknown): RequestInit => body === undefined
  ? { method: "POST" }
  : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", {}, signal),
  runPipeline: (input: PipelineRunRequest, signal?: AbortSignal) => request<PipelineJobStatus>("/pipeline/run", jsonPost(input), signal),
  getJob: (jobId: string, signal?: AbortSignal) => request<PipelineJobStatus>(`/pipeline/status/${encodeURIComponent(jobId)}`, {}, signal),
  runSync: (input: PipelineRunRequest, signal?: AbortSignal) => request<PipelineResult>("/pipeline/run_sync", jsonPost(input), signal),
  replayEvents: (signal?: AbortSignal) => request<ReplayEvents>("/replay/events", {}, signal),
  runReplay: (eventKey: string, signal?: AbortSignal) => request<PipelineJobStatus>(`/replay/${encodeURIComponent(eventKey)}`, jsonPost(), signal),
  runAmphanReplay: (signal?: AbortSignal) => request<AmphanOverview>("/api/amphan/replay", jsonPost(), signal),
  amphanOverview: (signal?: AbortSignal) => request<AmphanOverview>("/api/amphan/overview", {}, signal),
  amphanField: (variable: string, timeIndex = 0, signal?: AbortSignal, visualization = false, maxPoints?: number) => {
    const params = new URLSearchParams({ variable, time_index: String(timeIndex), visualization: String(visualization) });
    if (maxPoints !== undefined) params.set("max_points", String(maxPoints));
    return request<AmphanField>(`/api/amphan/field?${params.toString()}`, {}, signal);
  },
  amphanAlert: (timeIndex = 0, signal?: AbortSignal) =>
    request<AmphanAlert>(`/api/amphan/alert?time_index=${timeIndex}`, {}, signal),
  amphanTracks: (timeIndex = 0, signal?: AbortSignal) =>
    request<AmphanTracksResponse>(`/api/amphan/tracks?time_index=${timeIndex}`, {}, signal),
  runTiggeAmphanReplay: (signal?: AbortSignal) =>
    request<TiggeOverview>("/api/tigge/amphan/replay", jsonPost(), signal),
  tiggeAmphanOverview: (signal?: AbortSignal) =>
    request<TiggeOverview>("/api/tigge/amphan/overview", {}, signal),
  tiggeAmphanField: (variable: string, timeIndex = 0, signal?: AbortSignal, visualization = true, maxPoints?: number) => {
    const params = new URLSearchParams({ variable, time_index: String(timeIndex), visualization: String(visualization) });
    if (maxPoints !== undefined) params.set("max_points", String(maxPoints));
    return request<AmphanField>(`/api/tigge/amphan/field?${params.toString()}`, {}, signal);
  },
  tiggeAmphanAlert: (timeIndex = 0, signal?: AbortSignal) =>
    request<TiggeAlert>(`/api/tigge/amphan/alert?time_index=${timeIndex}`, {}, signal),
  tiggeAmphanTracks: (timeIndex = 0, signal?: AbortSignal) =>
    request<TiggeTracksResponse>(`/api/tigge/amphan/tracks?time_index=${timeIndex}`, {}, signal),
  forecastStatus: (signal?: AbortSignal) =>
    request<Record<string, unknown>>("/api/v1/forecast/status", {}, signal),
  forecastLatest: (signal?: AbortSignal) =>
    request<TiggeOverview>("/api/v1/forecast/latest", {}, signal),
  forecastField: (variable: string, timeIndex = 0, signal?: AbortSignal, maxPoints = 64) =>
    request<AmphanField>(`/api/v1/forecast/field?variable=${encodeURIComponent(variable)}&time_index=${timeIndex}&visualization=true&max_points=${maxPoints}`, {}, signal),
  forecastTracks: (timeIndex = 0, signal?: AbortSignal) =>
    request<TiggeTracksResponse>(`/api/v1/tracking/current?time_index=${timeIndex}`, {}, signal),
  forecastAlert: (timeIndex = 0, signal?: AbortSignal) =>
    request<TiggeAlert>(`/api/v1/alerts/current?time_index=${timeIndex}`, {}, signal),
  alerts: (query: AlertsQuery = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams();
    if (query.limit) params.set("limit", String(query.limit));
    if (query.severity) params.set("severity", query.severity);
    if (query.event_name) params.set("event_name", query.event_name);
    const suffix = params.size ? `?${params.toString()}` : "";
    return request<Alert[]>(`/alerts/latest${suffix}`, {}, signal);
  },
  alert: (alertId: number, signal?: AbortSignal) => request<Alert>(`/alerts/${alertId}`, {}, signal),
  metrics: (signal?: AbortSignal) => request<Metrics>("/metrics", {}, signal),
};