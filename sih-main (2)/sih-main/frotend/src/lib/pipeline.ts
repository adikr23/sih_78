import type { PipelineResult, Severity } from "./types";

export const PIPELINE_STAGES = [
  { id: 1, title: "Data ingestion", detail: "Ensemble + climatology", match: "generating ensemble" },
  { id: 2, title: "Spherical mesh", detail: "Icosahedral projection", match: "spherical mesh" },
  { id: 3, title: "EFI analysis", detail: "30-year baseline comparison", match: "computing EFI" },
  { id: 4, title: "GNN scan", detail: "Anomaly classification", match: "training GNN" },
  { id: 5, title: "4D tracking", detail: "Temporal trajectory linking", match: "4D bounding box" },
  { id: 6, title: "Diffusion", detail: "Experimental model-derived spatial refinement", match: "downscaling" },
  { id: 7, title: "Physics check", detail: "Conservation consistency", match: "physics consistency" },
  { id: 8, title: "Alert engine", detail: "Centroid + severity", match: "severity-tiered" },
] as const;

export function stageFromProgress(progress: string) {
  const direct = Number.parseInt(progress.split("/")[0] || "", 10);
  if (Number.isFinite(direct) && direct >= 1 && direct <= 8) return direct;
  return PIPELINE_STAGES.find((stage) => progress.includes(stage.match))?.id ?? 0;
}

export function severityLabel(severity?: Severity) {
  return severity ? severity.toUpperCase() : "NO RESULT";
}

export function physicsStatus(score?: number) {
  if (score === undefined) return "AWAITING RESULT";
  if (score >= 0.7) return "CONSISTENT";
  if (score >= 0.4) return "REVIEW";
  return "LOW CONFIDENCE";
}

export function resultMetrics(result?: PipelineResult) {
  return [
    ["EFI peak", result?.efi_peak],
    ["Diffusion peak", result?.diffusion_peak],
    ["U-Net peak", result?.unet_peak],
    ["Amplitude gain", result ? `${result.amplitude_gain_pct.toFixed(1)}%` : undefined],
    ["Physics score", result?.physics_score],
    ["Tracked points", result?.bounding_box.n_points],
    ["Forecast window", result ? `D+${result.bounding_box.t_start + 3} → D+${result.bounding_box.t_end + 3}` : undefined],
    ["Processing time", result ? `${result.elapsed_seconds.toFixed(1)}s` : undefined],
  ] as const;
}