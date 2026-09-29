export type JobStatus = "queued" | "running" | "done" | "failed";
export type Severity = "low" | "moderate" | "severe";
export type OperatingMode = "live" | "replay";

export interface HealthResponse {
  status: string;
  version: string;
  backend?: string;
  forecast_source?: string;
  forecast_status?: string;
  forecast_data?: "real" | "not_available" | string;
  data_mode?: string;
  ensemble_members?: number;
  gnn_status?: string;
  tracking_status?: string;
  downscaling_status?: string;
  verification_status?: string;
  mfc_status?: string;
  data_provider?: string;
  data_source?: string;
  source_label?: string;
  source_description?: string;
}

export interface PipelineRunRequest {
  event_name: string;
  seed: number | null;
}

export interface BoundingBox4D {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  z_min: number;
  z_max: number;
  t_start: number;
  t_end: number;
  n_points: number;
}

export interface PipelineResult {
  alert_id: number;
  centroid_row: number;
  centroid_col: number;
  peak_value: number;
  severity: Severity;
  confidence: number;
  radius_km: number;
  raw_score: number;
  physics_score: number;
  efi_peak: number;
  unet_peak: number;
  diffusion_peak: number;
  amplitude_gain_pct: number;
  bounding_box: BoundingBox4D;
  n_tracks: number;
  gnn_loss_history: number[];
  seed: number;
  elapsed_seconds: number;
  source_provider?: string;
  source_label?: string;
  source_description?: string;
  data_source?: string;
  latitude?: number;
  longitude?: number;
}

export interface PipelineJobStatus {
  job_id: string;
  status: JobStatus;
  progress: string;
  result: PipelineResult | null;
  error: string | null;
}

export interface ReplayEvents {
  available_events: string[];
}

export interface AmphanTimelineItem {
  time: string;
  efi_max: number;
  efi_min: number;
  anomaly_coverage: number;
  gnn_probability_max: number;
  active_nodes: number;
  track_count: number;
  lead_time_hours?: number;
  anomaly_max?: number;
}

export interface AmphanTrackPoint {
  valid_time: string;
  latitude: number;
  longitude: number;
  efi: number | null;
  confidence: number | null;
  severity: string | null;
  geographic_bbox: AmphanGeographicBounds;
  lead_time_hours?: number | null;
  anomaly_score?: number | null;
}

export interface AmphanGeographicBounds {
  latitude_min: number;
  latitude_max: number;
  longitude_west: number;
  longitude_east: number;
  crosses_dateline?: boolean;
}

export interface AmphanSpatialBounds {
  latitude_min: number;
  latitude_max: number;
  longitude_min: number;
  longitude_max: number;
}

export interface AmphanFieldStatistics {
  minimum: number;
  maximum: number;
  mean: number;
  p05: number;
  p50: number;
  p95: number;
}

export interface AmphanDemoSummary {
  project: string;
  event: string;
  event_period: string;
  dataset: string;
  reference_period: string;
  spatial_bounds: AmphanSpatialBounds | null;
  grid_dimensions: { latitude: number; longitude: number } | null;
  number_of_timestamps: number;
  processing_stages: string[];
  alert: AmphanOverview["alert"];
  physics_status: string;
  verification_status: string;
  rainfall_status: string;
  data_limitations: string[];
  model_comparison: {
    classification: string;
    units: string;
    independent_verification: string;
    fields: Record<string, AmphanFieldStatistics>;
  };
}

export interface AmphanAlert {
  event: string;
  time: string;
  location: { latitude: number; longitude: number };
  severity: string;
  severity_label: string;
  efi: number;
  confidence: number;
  confidence_status: string;
  status: string;
  operational_warning: false;
  track: AmphanTrackPoint | null;
  bounding_box: AmphanGeographicBounds | null;
  notice: string;
}

export interface AmphanTrack {
  track_id: string;
  geographic_bbox: AmphanGeographicBounds;
  track_points: AmphanTrackPoint[];
  selected_track_point: AmphanTrackPoint;
}

export interface AmphanTracksResponse {
  event: string;
  time: string;
  active_tracks: AmphanTrack[];
  dominant_track: AmphanTrack | null;
}

export interface AmphanOverview {
  event: string;
  event_period: string;
  status: string;
  data_source: string;
  mode: string;
  progress: Array<{ stage: string; status: string; classification?: string; supervision?: string }>;
  alert: {
    event: string;
    event_period: string;
    centroid: { latitude: number; longitude: number };
    severity: string;
    severity_label: string;
    radius_km: number | null;
    efi: number;
    track_id: string | null;
    track_available: boolean;
    data_source: string;
    mode: string;
    provenance: Record<string, unknown>;
  };
  timeline: AmphanTimelineItem[];
  tracks: Array<Record<string, unknown>>;
  dominant_track: { track_id?: string; track_points?: AmphanTrackPoint[] } | null;
  downscaling: {
    status: string;
    label: string;
    reference_target_available: boolean;
    model_to_model_diagnostic: Record<string, number | string | boolean>;
  };
  physics: {
    physics_score: number;
    physics_status: string;
    physics_method: string;
    mfc_available: boolean;
    reason: string;
  };
  verification: {
    verification_status: string;
    reason: string;
    stage1_descriptive_metrics: Record<string, unknown>;
    downscaling_metrics: Record<string, unknown>;
    model_to_model_diagnostic: Record<string, unknown>;
  };
  data_status: Record<string, unknown>;
  provenance: Record<string, unknown>;
  demo_summary: AmphanDemoSummary;
}

export interface AmphanField {
  variable: string;
  status: string;
  valid_time?: string;
  lead_time_hours?: number;
  units?: string | null;
  latitude: number[];
  longitude: number[];
  values: number[][];
  provenance?: Record<string, unknown>;
  classification?: string;
  reason?: string;
  available?: boolean;
  visualization_downsampled?: boolean;
  visualization_only?: boolean;
  source_shape?: number[];
  returned_shape?: number[];
  statistics?: AmphanFieldStatistics;
}

export interface TiggeTimelineItem {
  time: string;
  valid_time: string;
  lead_time_hours: number;
  efi_max: number;
  anomaly_max: number;
  anomaly_coverage?: number;
  gnn_probability_max: number;
  active_nodes: number;
  track_count: number;
}

export interface TiggeAlert {
  event: string;
  time: string;
  location: { latitude: number; longitude: number };
  severity: string;
  confidence: number;
  anomaly_score: number;
  temperature_efi_like: number;
  lead_time_hours: number;
  source: string;
  model_version: string;
  notice: string;
  track: AmphanTrackPoint | null;
  bounding_box: AmphanGeographicBounds | null;
}

export interface TiggeOverview {
  event: string;
  event_period: string;
  status: string;
  data_source: string;
  mode: string;
  forecast_initialization_time: string;
  progress: Array<{ stage: string; status: string }>;
  timeline: TiggeTimelineItem[];
  tracks: Array<{ track_id: string; track_points: AmphanTrackPoint[] }>;
  dominant_track: { track_id?: string; track_points?: AmphanTrackPoint[] } | null;
  alert: TiggeAlert & { centroid: { latitude: number; longitude: number } };
  spatial_bounds: AmphanSpatialBounds;
  grid_dimensions: { latitude: number; longitude: number };
  physics: { physics_status: string; physics_method: string; reason: string };
  verification: {
    verification_status: string;
    reason: string;
    diagnostics: Record<string, unknown>;
  };
  model_status: {
    classification: string;
    model_version: string;
    training_epochs: number;
    track_probability_threshold?: number;
    calibrated: boolean;
    accuracy: string;
  };
  data_status: Record<string, unknown>;
  provenance: Record<string, unknown>;
  anomaly_classification: string;
  data_limitations: string[];
  demo_summary: Record<string, unknown>;
}

export interface TiggeTracksResponse {
  event: string;
  time: string;
  active_tracks: Array<{
    track_id: string;
    geographic_bbox: AmphanGeographicBounds;
    track_points: AmphanTrackPoint[];
    selected_track_point: AmphanTrackPoint;
  }>;
  dominant_track: {
    track_id: string;
    track_points: AmphanTrackPoint[];
    selected_track_point: AmphanTrackPoint;
  } | null;
}

export interface Alert {
  id: number;
  event_name: string;
  centroid_row: number;
  centroid_col: number;
  peak_value: number;
  severity: Severity;
  confidence: number;
  radius_km: number;
  physics_score: number;
  efi_peak: number;
  unet_peak: number;
  diffusion_peak: number;
  amplitude_gain_pct: number;
  source_provider?: string;
  source_label?: string;
  source_description?: string;
  data_source?: string;
  created_at: string | null;
  latitude?: number;
  longitude?: number;
}

export interface Metrics {
  total_alerts: number;
  by_severity: Record<Severity, number>;
  avg_physics_score: number;
  avg_diffusion_vs_unet_amplitude_gain_pct: number;
}

export interface AlertsQuery {
  limit?: number;
  severity?: Severity;
  event_name?: string;
}

export interface ApiErrorShape {
  detail?: string;
}

export interface JobRecord {
  id: string;
  mode: OperatingMode;
  eventName: string;
  startedAt: string;
  completedAt?: string;
  status: JobStatus;
  result?: PipelineResult;
}