import { Fragment, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useEffect } from "react";
import { AlertCircle, ArrowRight, Braces, CheckCircle2, Clipboard, CloudOff, Database, LoaderCircle, Pause, Play, RefreshCw, Search, ShieldAlert, SkipBack, SkipForward, Sparkles } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from "recharts";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PageHeading } from "@/components/layout/app-shell";
import { PipelineTimeline } from "@/components/science/pipeline-timeline";
import { ForecastTimeline, MetricTooltip, PhysicsGauge, ThreatPanel } from "@/components/science/visualizations";
import { useStormtrace } from "@/components/stormtrace-provider";
import { useAlert, useAlerts, useBackendHealth, useMetrics, useSyncPipeline } from "@/hooks/use-stormtrace";
import { API_BASE_URL, api } from "@/lib/api";
import { resultMetrics, severityLabel } from "@/lib/pipeline";
import type { Alert, AmphanAlert, AmphanField, AmphanGeographicBounds, AmphanOverview, AmphanTrackPoint, Severity, TiggeOverview } from "@/lib/types";

const getSourceLabel = (source?: string) => source || "public-weather-forecast";
const displayEvent = (event: string) => event.replaceAll("_", " ");

export function CommandCenterPage() {
  const health = useBackendHealth();
  const storm = useStormtrace();
  const [timeIndex, setTimeIndex] = useState(0);
  const [fieldTimeIndex, setFieldTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [mounted, setMounted] = useState(false);
  const queryClient = useQueryClient();
  useEffect(() => setMounted(true), []);
  const result = storm.currentResult;
  const eventName = storm.jobs[0]?.eventName;
  const sourceLabel = mounted
    ? getSourceLabel(health.data?.source_label ?? health.data?.data_source)
    : "Forecast source";
  const forecastStatusQuery = useQuery({
    queryKey: ["forecast-status"],
    queryFn: ({ signal }) => api.forecastStatus(signal),
    refetchInterval: 60_000,
    retry: 1,
  });
  const tiggeOverviewQuery = useQuery({
    queryKey: ["forecast-latest"],
    queryFn: ({ signal }) => api.forecastLatest(signal),
    staleTime: 60_000,
    refetchInterval: 60_000,
    retry: 1,
  });
  const tigge = tiggeOverviewQuery.data;
  const selectedTime = tigge?.timeline[timeIndex];
  useEffect(() => {
    if (!tigge) return;
    const cacheKey = ["forecast-field", tigge.forecast_initialization_time, "t2m", timeIndex];
    if (queryClient.getQueryData(cacheKey)) {
      setFieldTimeIndex(timeIndex);
      return;
    }
    const timer = window.setTimeout(() => setFieldTimeIndex(timeIndex), 100);
    return () => window.clearTimeout(timer);
  }, [queryClient, tigge, timeIndex]);
  useEffect(() => {
    if (!isPlaying || !tigge?.timeline.length) return;
    const timer = window.setInterval(() => {
      setTimeIndex((current) => current >= tigge.timeline.length - 1 ? 0 : current + 1);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [isPlaying, tigge?.timeline.length]);
  const fieldQuery = useQuery({
    queryKey: ["forecast-field", tigge?.forecast_initialization_time, "t2m", fieldTimeIndex],
    queryFn: ({ signal }) => api.forecastField("t2m", fieldTimeIndex, signal, 64),
    enabled: Boolean(tigge),
    staleTime: Infinity,
    gcTime: 5 * 60_000,
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === tigge?.forecast_initialization_time ? previous : undefined,
    retry: false,
  });
  const tracksQuery = useQuery({
    queryKey: ["forecast-tracks", fieldTimeIndex],
    queryFn: ({ signal }) => api.forecastTracks(fieldTimeIndex, signal),
    enabled: Boolean(tigge),
    staleTime: 60_000,
    retry: false,
  });
  const alertQuery = useQuery({
    queryKey: ["forecast-alert", fieldTimeIndex],
    queryFn: ({ signal }) => api.forecastAlert(fieldTimeIndex, signal),
    enabled: Boolean(tigge),
    staleTime: 60_000,
    retry: false,
  });
  const activeTrack = tracksQuery.data?.dominant_track ?? tigge?.dominant_track ?? null;
  const displayedTime = tigge?.timeline.find((item) => item.valid_time === fieldQuery.data?.valid_time);
  const trackPoints = (activeTrack?.track_points ?? []).filter((point) =>
    point.valid_time <= (displayedTime?.valid_time ?? ""),
  );
  const selectedTrackPoint = activeTrack?.track_points?.find(
    (point) => point.valid_time === displayedTime?.valid_time,
  );
  const alertLocation = alertQuery.data && alertQuery.data.time === displayedTime?.valid_time ? alertQuery.data.location : undefined;
  const center = alertLocation
    && Number.isFinite(alertLocation.latitude)
    && Number.isFinite(alertLocation.longitude)
    ? alertLocation
    : selectedTrackPoint
      ? { latitude: selectedTrackPoint.latitude, longitude: selectedTrackPoint.longitude }
      : undefined;
  const trackBounds = selectedTrackPoint?.geographic_bbox;
  useEffect(() => {
    if (!tigge) return;
    const timer = window.setTimeout(() => {
    const candidates = [timeIndex - 2, timeIndex - 1, timeIndex + 1, timeIndex + 2]
      .filter((index) => index >= 0 && index < tigge.timeline.length);
    for (const index of candidates) {
      void queryClient.prefetchQuery({
        queryKey: ["forecast-field", tigge.forecast_initialization_time, "t2m", index],
        queryFn: ({ signal }) => api.forecastField("t2m", index, signal, 64),
        staleTime: Infinity,
        gcTime: 5 * 60_000,
      });
    }
    }, 100);
    return () => window.clearTimeout(timer);
  }, [queryClient, tigge, timeIndex]);
  return <><PageHeading eyebrow="Operational overview" title="Command Center" description="Spherical anomaly tracking, amplitude-preserving downscaling and physics-aware alerting in one auditable view." action={<div className="flex gap-2"><Button variant="outline" asChild><Link to="/historical-replay">Replay cases</Link></Button><Button onClick={() => storm.startReplay()} disabled={storm.isStarting}><Play /> Run demonstration</Button></div>} />
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border border-primary/20 bg-primary/5 px-4 py-3"><div><span className="mr-3 text-[10px] font-bold uppercase text-primary">Current forecast state</span><span className="text-xs text-muted-foreground">{!mounted ? "Loading forecast status…" : forecastStatusQuery.data?.["stale"] ? "FORECAST STALE · showing last valid local cycle" : "Current local forecast cycle"}</span></div><div className="flex flex-wrap gap-3 text-[10px] uppercase"><span className="data-mode">Source: {sourceLabel}</span><span className="data-mode">Mode: {mounted ? String(forecastStatusQuery.data?.["mode"] ?? health.data?.data_mode ?? "UNKNOWN") : "Loading"}</span><span className="data-mode">{mounted ? `${String(forecastStatusQuery.data?.["ensemble_members"] ?? 0)} members · ${String(forecastStatusQuery.data?.["available_steps"] ?? 0)} steps` : "Loading forecast dimensions"}</span></div></div>
    {storm.currentJob && <div className="mb-4"><PipelineTimeline job={storm.currentJob} onRetry={() => storm.mode === "replay" ? storm.startReplay() : storm.startLive()} /></div>}
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.8fr)_minmax(290px,.7fr)]">
      <section className="panel min-w-0 p-4" aria-label="Geospatial operations view">
        <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="data-label">Geospatial operations view</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Data source: NCMRWF TIGGE · Variable: 2 m temperature · Resolution: ~12 km source · Status: REAL DATA
            </div>
          </div>
          <span className="data-mode">NCMRWF TIGGE · REAL FORECAST</span>
        </div>
        {tiggeOverviewQuery.isPending && <div className="grid min-h-[420px] place-items-center rounded border border-border text-sm text-muted-foreground">Loading geospatial forecast…</div>}
        {tiggeOverviewQuery.isError && <div role="alert" className="grid min-h-[420px] place-items-center rounded border border-destructive/40 p-6 text-center">
          <div><div className="font-semibold text-destructive">Geospatial data unavailable</div><p className="mt-2 text-sm text-muted-foreground">{tiggeOverviewQuery.error.message}</p><Button className="mt-3" variant="outline" onClick={() => void tiggeOverviewQuery.refetch()}>Retry</Button></div>
        </div>}
        {tigge && <>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div className="font-mono text-xs">{selectedTime?.valid_time ?? "No forecast time"} · lead +{selectedTime?.lead_time_hours ?? "—"} h</div>
            <div className="text-xs text-muted-foreground">
              {alertQuery.data ? `${alertQuery.data.severity} · confidence ${(alertQuery.data.confidence * 100).toFixed(1)}% · experimental` : "Derived alert unavailable"}
            </div>
            {fieldQuery.data && fieldQuery.data.valid_time !== selectedTime?.valid_time && <div className="mb-2 text-[10px] text-muted-foreground" aria-live="polite">Loading {selectedTime?.valid_time} · retaining the previously rendered field and overlays until the selected timestep is ready.</div>}
          </div>
          <div className="mb-3 flex items-center gap-2">
            <Button variant="outline" size="icon" aria-label="Previous forecast time" onClick={() => setTimeIndex((current) => Math.max(0, current - 1))}><SkipBack /></Button>
            <Button variant="outline" size="icon" aria-label={isPlaying ? "Pause forecast timeline" : "Play forecast timeline"} onClick={() => setIsPlaying((value) => !value)}>{isPlaying ? <Pause /> : <Play />}</Button>
            <Button variant="outline" size="icon" aria-label="Next forecast time" onClick={() => setTimeIndex((current) => Math.min(tigge.timeline.length - 1, current + 1))}><SkipForward /></Button>
            <input
              aria-label="Command Center forecast time"
              className="w-full accent-primary"
              type="range"
              min={0}
              max={Math.max(0, tigge.timeline.length - 1)}
              value={Math.min(timeIndex, Math.max(0, tigge.timeline.length - 1))}
              onChange={(event) => setTimeIndex(Number(event.target.value))}
            />
          </div>
          {fieldQuery.isPending && !fieldQuery.data && <div className="grid min-h-[420px] place-items-center rounded border border-border text-sm text-muted-foreground">Loading geospatial forecast…</div>}
          {fieldQuery.isError && <div role="alert" className="grid min-h-[420px] place-items-center rounded border border-destructive/40 p-6 text-center">
            <div><div className="font-semibold text-destructive">Geospatial data unavailable</div><p className="mt-2 text-sm text-muted-foreground">{fieldQuery.error.message}</p></div>
          </div>}
          {fieldQuery.data && fieldQuery.data.values.length > 0
            ? <GeographicField
              field={fieldQuery.data}
              title="TIGGE 2 m temperature · ensemble mean"
              trackPoints={trackPoints}
              center={center}
              bounds={trackBounds}
              palette="temperature"
              geographicSource="NCMRWF TIGGE"
            />
            : !fieldQuery.isPending && !fieldQuery.isError && <div className="grid min-h-[420px] place-items-center rounded border border-border text-sm text-muted-foreground">No geospatial field available for this timestep.</div>}
          <p className="mt-2 text-[10px] text-muted-foreground">
            {fieldQuery.data?.visualization_downsampled
              ? `Visualization only: ${fieldQuery.data.source_shape?.join("×") ?? "source grid"} sampled to ${fieldQuery.data.returned_shape?.join("×") ?? "display grid"}. `
              : ""}
            {alertQuery.isError ? "Alert overlay unavailable. " : ""}
            {tracksQuery.isError ? "Track overlay unavailable. " : ""}
            Derived track and alert are experimental and are not operational warnings.
          </p>
        </>}
      </section>
      <ThreatPanel result={result} eventName={eventName} />
    </div>
    <div className="mt-4 grid gap-px overflow-hidden border border-border bg-border grid-cols-2 md:grid-cols-4 xl:grid-cols-8">{resultMetrics(result).map(([label,value]) => <MetricTooltip key={label} label={label} value={value} description={metricDescription(label)} />)}</div>
    <div className="mt-4 grid gap-4 xl:grid-cols-[1.3fr_.7fr]"><ForecastTimeline result={result} /><PhysicsGauge result={result} /></div>
  </>;
}

export function LiveAnalysisPage() {
  const health = useBackendHealth();
  const storm = useStormtrace();
  const sourceLabel = getSourceLabel(health.data?.source_label ?? health.data?.data_source);
  return <><PageHeading eyebrow="Asynchronous pipeline" title="Live Analysis" description="Start a fresh forecast run with the configured ingestion provider; the backend uses the active source label for traceability." action={<Button onClick={storm.startLive} disabled={storm.isStarting}>{storm.isStarting ? <LoaderCircle className="animate-spin" /> : <Play />} Run live analysis</Button>} /><HonestyNote sourceLabel={sourceLabel} /><PipelineTimeline job={storm.currentJob} onRetry={storm.startLive} />{storm.currentResult && <div className="mt-4 grid gap-4 lg:grid-cols-2"><ThreatPanel result={storm.currentResult} eventName="live_run" /><JsonPanel value={storm.currentResult} /></div>}</>;
}

export function ReplayPage() {
  const queryClient = useQueryClient();
  const [timeIndex, setTimeIndex] = useState(0);
  const health = useBackendHealth();
  const runReplay = useMutation({
    mutationFn: ({ signal }: { signal?: AbortSignal }) => api.runAmphanReplay(signal),
    onSuccess: (result) => {
      queryClient.setQueryData(["amphan-overview"], result);
      void health.refetch();
      setTimeIndex(0);
    },
    onError: (error) => toast.error("REPLAY NOT READY", { description: error.message }),
  });
  const replay = runReplay.data;
  const t2m = useQuery({
    queryKey: ["amphan-field", "t2m", timeIndex],
    queryFn: ({ signal }) => api.amphanField("t2m", timeIndex, signal, true),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const efi = useQuery({
    queryKey: ["amphan-field", "efi", timeIndex],
    queryFn: ({ signal }) => api.amphanField("efi", timeIndex, signal, true),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const gnn = useQuery({
    queryKey: ["amphan-field", "gnn_probability", timeIndex],
    queryFn: ({ signal }) => api.amphanField("gnn_probability", timeIndex, signal, true),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const coarse = useQuery({
    queryKey: ["amphan-field", "crop_t2m"],
    queryFn: ({ signal }) => api.amphanField("crop_t2m", 0, signal, true, 32),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const unet = useQuery({
    queryKey: ["amphan-field", "unet"],
    queryFn: ({ signal }) => api.amphanField("unet", 0, signal, true, 32),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const diffusion = useQuery({
    queryKey: ["amphan-field", "diffusion"],
    queryFn: ({ signal }) => api.amphanField("diffusion", 0, signal, true, 32),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const alert = useQuery({
    queryKey: ["amphan-alert", timeIndex],
    queryFn: ({ signal }) => api.amphanAlert(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const tracks = useQuery({
    queryKey: ["amphan-tracks", timeIndex],
    queryFn: ({ signal }) => api.amphanTracks(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: 60_000,
  });
  const currentTime = replay?.timeline[timeIndex]?.time;
  const trackPoints = tracks.data?.dominant_track?.track_points ?? [];
  const currentTrackPoint = tracks.data?.dominant_track?.selected_track_point;
  const currentCenter = alert.data?.location ?? replay?.alert.centroid;
  const currentBounds = alert.data?.bounding_box ?? undefined;
  const isBackendOffline = !health.isSuccess;
  return <>
    <PageHeading eyebrow="IMDAA historical replay" title="Cyclone Amphan · May 16–21, 2020" description="Cached historical reanalysis replay. This is not an operational forecast or independent observational verification." action={<span className="data-mode">IMDAA · HISTORICAL_REPLAY</span>} />
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border border-primary/20 bg-primary/5 px-4 py-3">
      <div><span className="mr-3 text-[10px] font-bold uppercase text-primary">Data source</span><span className="text-xs text-muted-foreground">IMDAA regional reanalysis · May 2011–2019 reference · May 16–21, 2020 event holdout</span></div>
      <Button onClick={() => runReplay.mutate({})} disabled={runReplay.isPending}>
        {runReplay.isPending ? <LoaderCircle className="animate-spin" /> : <Play />}
        Run Amphan Replay
      </Button>
    </div>
    {isBackendOffline && <section role="alert" className="panel mb-4 border border-destructive/40 p-4 text-sm">
      <div className="font-semibold text-destructive">BACKEND OFFLINE</div>
      <p className="mt-2 text-muted-foreground">{health.error instanceof Error ? `Health check failed: ${health.error.message}. ` : "The backend health check has not succeeded. "}Start the backend from its directory with <code>python -m uvicorn src.main:app --host 127.0.0.1 --port 8000</code>, then retry.</p>
    </section>}
    {runReplay.isError && !replay && !isBackendOffline && <section role="alert" className="panel mb-4 border border-amber-500/30 p-4 text-sm">
      <div className="font-semibold">Replay cache is not prepared or unavailable</div>
      <p className="mt-2 text-muted-foreground">Build the IMDAA regional reference, event, and replay caches from the backend directory:</p>
      <code className="mt-2 block overflow-x-auto rounded bg-muted p-3 text-xs">.\.venv\Scripts\python scripts\build_imdaa_cache.py</code>
      <p className="mt-2 text-xs text-muted-foreground">{runReplay.error instanceof Error ? runReplay.error.message : "The backend has not prepared the historical replay."}</p>
    </section>}
    {replay && <>
      <ReplayPipelineCard stages={replay.demo_summary.processing_stages} />
      <div className="panel mb-4 p-4">
        <div className="flex items-center justify-between gap-2">
          <div><div className="data-label">Cyclone Amphan</div><div className="mt-1 text-sm font-semibold">May 16–21, 2020 · historical replay</div></div>
          <div className="data-mode">{replay.timeline.length} cached timestamps</div>
        </div>
        <div className="mt-4 text-[10px] uppercase tracking-wide text-muted-foreground">Processing status</div>
        <div className="mt-3 grid gap-2 sm:grid-cols-4 xl:grid-cols-7">
          {replay.progress.map((stage) => <div key={stage.stage} className="metric-mini"><span>{stage.stage}</span><strong>{stage.status.replaceAll("_", " ")}</strong></div>)}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <label className="data-label" htmlFor="amphan-time">Event timeline</label>
          <input id="amphan-time" type="range" min={0} max={Math.max(0, replay.timeline.length - 1)} value={timeIndex} onChange={(event) => setTimeIndex(Number(event.target.value))} className="w-full accent-primary" />
          <span className="min-w-44 text-right font-mono text-xs">{currentTime ?? "No timeline"}</span>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <GeographicField field={t2m.data} title="Raw 2 m temperature (IMDAA)" trackPoints={trackPoints} center={currentCenter} bounds={currentBounds} palette="temperature" />
        <GeographicField field={efi.data} title="t2m EFI-like anomaly score" trackPoints={trackPoints} center={currentCenter} bounds={currentBounds} palette="efi" />
        <GeographicField field={gnn.data} title="GNN anomaly probability (weakly supervised)" trackPoints={trackPoints} center={currentCenter} bounds={currentBounds} palette="efi" />
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1fr]">
        <ReplaySummary replay={replay} alert={alert.data} trackPoint={currentTrackPoint} trackId={tracks.data?.dominant_track?.track_id} />
        <section className="panel p-4">
          <div className="data-label">Track at selected timestamp</div>
          {currentTrackPoint ? <>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="metric-mini"><span>Track ID</span><strong>{tracks.data?.dominant_track?.track_id ?? "Unavailable"}</strong></div>
              <div className="metric-mini"><span>Timestamp</span><strong>{currentTrackPoint.valid_time.replace("T", " ")}</strong></div>
              <div className="metric-mini"><span>Latitude</span><strong>{currentTrackPoint.latitude.toFixed(3)}°</strong></div>
              <div className="metric-mini"><span>Longitude</span><strong>{currentTrackPoint.longitude.toFixed(3)}°</strong></div>
              <div className="metric-mini"><span>EFI</span><strong>{currentTrackPoint.efi?.toFixed(3) ?? "N/A"}</strong></div>
              <div className="metric-mini"><span>Model probability</span><strong>{currentTrackPoint.confidence === null ? "N/A" : `${(currentTrackPoint.confidence * 100).toFixed(1)}%`}</strong></div>
              <div className="metric-mini"><span>Severity</span><strong>{alert.data?.severity ?? "N/A"}</strong></div>
              <div className="metric-mini col-span-2"><span>Geographic bounding box</span><strong>{formatBounds(currentTrackPoint.geographic_bbox)}</strong></div>
            </div>
            <p className="mt-3 text-[10px] text-muted-foreground">Coordinates and bounding box are from the cached tracked mesh cluster; confidence is an uncalibrated model score.</p>
          </> : <p className="mt-3 text-sm text-muted-foreground">No tracked cluster is available at this timestamp; no location is inferred.</p>}
        </section>
      </div>
      <ModelComparison coarse={coarse.data} unet={unet.data} diffusion={diffusion.data} />
      <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1fr]">
        <section className="panel p-4">
          <div className="data-label">Physics & verification</div>
          <div className="mt-3 font-semibold">{replay.physics.physics_status}</div>
          <p className="mt-1 text-xs text-muted-foreground">{replay.physics.physics_method}: {replay.physics.reason}</p>
          <div className="mt-4 font-semibold">{replay.verification.verification_status}</div>
          <p className="mt-1 text-xs text-muted-foreground">{replay.verification.reason}</p>
          <p className="mt-3 text-xs text-amber-300">Independent high-resolution verification unavailable.</p>
        </section>
        <DataIntegrityPanel replay={replay} />
      </div>
      <div className="mt-4 panel p-4">
        <div className="data-label">Timeline · valid time / EFI / GNN</div>
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
          {replay.timeline.map((item, index) => <button key={item.time} onClick={() => setTimeIndex(index)} className={`metric-mini text-left ${index === timeIndex ? "border-primary bg-primary/10" : ""}`}>
            <span>{item.time.replace("T", " ")}</span><strong>EFI {item.efi_max.toFixed(2)}</strong><span>coverage {(item.anomaly_coverage * 100).toFixed(1)}%</span>
          </button>)}
        </div>
      </div>
    </>}
  </>;
}

export function TiggeReplayPage() {
  const queryClient = useQueryClient();
  const [timeIndex, setTimeIndex] = useState(0);
  const health = useBackendHealth();
  const replayQuery = useQuery({
    queryKey: ["tigge-amphan-overview"],
    queryFn: ({ signal }) => api.tiggeAmphanOverview(signal),
    enabled: false,
    staleTime: Infinity,
    retry: false,
  });
  const runReplay = useMutation({
    mutationFn: ({ signal }: { signal?: AbortSignal }) => api.runTiggeAmphanReplay(signal),
    onSuccess: (result) => {
      queryClient.setQueryData(["tigge-amphan-overview"], result);
      setTimeIndex(0);
    },
    onError: (error) => toast.error("TIGGE REPLAY NOT READY", { description: error.message }),
  });
  const replay = runReplay.data ?? replayQuery.data;
  const fieldQuery = (variable: string) => ({
    queryKey: ["tigge-amphan-field", variable, timeIndex],
    queryFn: ({ signal }: { signal: AbortSignal }) => api.tiggeAmphanField(variable, timeIndex, signal, true),
    enabled: Boolean(replay),
    staleTime: Infinity,
  });
  const t2m = useQuery(fieldQuery("t2m"));
  const anomaly = useQuery(fieldQuery("anomaly_score"));
  const gnn = useQuery(fieldQuery("gnn_probability"));
  const alert = useQuery({
    queryKey: ["tigge-amphan-alert", timeIndex],
    queryFn: ({ signal }) => api.tiggeAmphanAlert(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: Infinity,
  });
  const tracks = useQuery({
    queryKey: ["tigge-amphan-tracks", timeIndex],
    queryFn: ({ signal }) => api.tiggeAmphanTracks(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: Infinity,
  });
  const currentTime = replay?.timeline[timeIndex]?.valid_time;
  const track = tracks.data?.dominant_track;
  const selectedPoint = track?.selected_track_point;
  const trackPoints = track?.track_points ?? [];
  const isBackendOffline = !health.isSuccess;
  const forecastProvenance = replay?.provenance["forecast"];
  const forecastFile = forecastProvenance && typeof forecastProvenance === "object"
    ? String((forecastProvenance as Record<string, unknown>)["source_file"] ?? "")
    : "";
  const forecastMetadata = forecastProvenance && typeof forecastProvenance === "object"
    ? forecastProvenance as Record<string, unknown>
    : {};
  const forecastMembers = String(forecastMetadata["ensemble_count"] ?? "—");
  const forecastSteps = replay?.timeline.length ?? 0;
  const forecastDomain = forecastMetadata["region"] && typeof forecastMetadata["region"] === "object"
    ? JSON.stringify(forecastMetadata["region"])
    : "configured analysis domain";
  const startBackendReplay = () => runReplay.mutate({});
  return <>
    <PageHeading
      eyebrow="NCMRWF TIGGE forecast replay"
      title="Cyclone Amphan · Real Forecast"
      description="Cached forecast initialization replay with experimental anomaly scoring and unverified geographic tracks."
      action={<span className="data-mode">NCMRWF TIGGE · REAL FORECAST</span>}
    />
    <div className="panel mb-4 flex flex-wrap items-center justify-between gap-3 p-4">
      <div>
        <div className="data-label">NCMRWF TIGGE · {replay?.forecast_initialization_time ?? "initialization unavailable"}</div>
        <div className="mt-1 text-xs text-muted-foreground">{forecastMembers} perturbed members · {forecastSteps} source valid times · {forecastDomain}</div>
      </div>
      <Button onClick={startBackendReplay} disabled={runReplay.isPending || Boolean(replay)}>
        {runReplay.isPending ? <LoaderCircle className="animate-spin" /> : <Play />}
        {replay ? "Forecast replay loaded" : "Run TIGGE Amphan Replay"}
      </Button>
    </div>
    {isBackendOffline && <section role="alert" className="panel mb-4 border border-destructive/40 p-4 text-sm">
      <div className="font-semibold text-destructive">BACKEND OFFLINE</div>
      <p className="mt-2 text-muted-foreground">Start the backend from its directory with <code>python -m uvicorn src.main:app --host 127.0.0.1 --port 8000</code>.</p>
    </section>}
    {runReplay.isError && !isBackendOffline && <section role="alert" className="panel mb-4 border border-amber-500/30 p-4 text-sm">
      <div className="font-semibold">TIGGE replay cache is not ready</div>
      <p className="mt-2 text-muted-foreground">Build it in the backend directory with:</p>
      <code className="mt-2 block overflow-x-auto rounded bg-muted p-3 text-xs">.\.venv\Scripts\python scripts\build_tigge_amphan_cache.py</code>
      <p className="mt-2 text-xs text-muted-foreground">{runReplay.error instanceof Error ? runReplay.error.message : "The precomputed forecast replay is unavailable."}</p>
      <Button className="mt-3" variant="outline" onClick={startBackendReplay}>Retry loading cache</Button>
    </section>}
    {replay && <>
      <section className="panel mb-4 p-4">
        <div className="flex items-center justify-between gap-3">
          <div><div className="data-label">{replay.event} · forecast initialization {replay.forecast_initialization_time}</div><div className="mt-1 text-xs text-muted-foreground">{forecastFile}</div><div className="mt-1 text-xs text-muted-foreground">Centre {String(forecastMetadata["centre"] ?? "NCMRWF")} · members {replay.data_status["forecast"] && typeof replay.data_status["forecast"] === "object" ? String((replay.data_status["forecast"] as Record<string, unknown>)["ensemble_count"] ?? "N/A") : "N/A"} · lead +{replay.timeline[timeIndex]?.lead_time_hours ?? 0}h</div></div>
          <div className="data-mode">{replay.grid_dimensions.latitude}×{replay.grid_dimensions.longitude} · {replay.timeline.length} valid times</div>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <label className="data-label shrink-0" htmlFor="tigge-time">Forecast lead</label>
          <input id="tigge-time" type="range" min={0} max={Math.max(0, replay.timeline.length - 1)} value={timeIndex} onChange={(event) => setTimeIndex(Number(event.target.value))} className="w-full accent-primary" />
          <span className="min-w-52 text-right font-mono text-xs">{currentTime ?? "No valid time"} · +{replay.timeline[timeIndex]?.lead_time_hours ?? 0}h</span>
        </div>
      </section>
      <section className="panel mb-4 p-4">
        <div className="data-label">Cached inference pipeline · no retraining on requests</div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {replay.progress.map((stage, index) => <div className="metric-mini" key={stage.stage}>
            <span>{String(index + 1).padStart(2, "0")} · {stage.stage}</span>
            <strong className="text-status-ok">{stage.status}</strong>
          </div>)}
        </div>
      </section>
      <div className="grid gap-4 xl:grid-cols-2">
        <GeographicField field={t2m.data} title="TIGGE 2 m temperature" trackPoints={trackPoints} center={alert.data?.location} bounds={alert.data?.bounding_box ?? undefined} palette="temperature" geographicSource="NCMRWF TIGGE" />
        <GeographicField field={anomaly.data} title="Temperature EFI-like + wind anomaly proxy" trackPoints={trackPoints} center={alert.data?.location} bounds={alert.data?.bounding_box ?? undefined} palette="efi" geographicSource="NCMRWF TIGGE" />
        <GeographicField field={gnn.data} title="Experimental weakly supervised GNN score" trackPoints={trackPoints} center={alert.data?.location} bounds={alert.data?.bounding_box ?? undefined} palette="efi" geographicSource="NCMRWF TIGGE" />
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <section className="panel border border-amber-500/30 p-4">
          <div className="data-label">STORMTRACE RESEARCH ALERT · NOT OPERATIONAL</div>
          {alert.data && <div className="mt-3 grid grid-cols-2 gap-3">
            <div className="metric-mini"><span>Event / valid time</span><strong>{alert.data.event} · {alert.data.time.replace("T", " ")}</strong></div>
            <div className="metric-mini"><span>Lead</span><strong>+{alert.data.lead_time_hours} h</strong></div>
            <div className="metric-mini"><span>Location</span><strong>{alert.data.location.latitude.toFixed(3)}°, {alert.data.location.longitude.toFixed(3)}°</strong></div>
            <div className="metric-mini"><span>Project tier · anomaly-area heuristic</span><strong>{alert.data.severity}</strong></div>
            <div className="metric-mini"><span>EFI-like anomaly</span><strong>{alert.data.temperature_efi_like.toFixed(3)}</strong></div>
            <div className="metric-mini"><span>GNN score</span><strong>{(alert.data.confidence * 100).toFixed(1)}% · uncalibrated</strong></div>
            <div className="metric-mini col-span-2"><span>Source / model version</span><strong>{alert.data.source} · {alert.data.model_version}</strong></div>
          </div>}
          <p className="mt-3 rounded border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200">{alert.data?.notice ?? "This is a research prototype replay, not an official warning."}</p>
        </section>
        <section className="panel p-4">
          <div className="data-label">Detected anomaly track · demonstration only</div>
          {selectedPoint ? <div className="mt-3 grid grid-cols-2 gap-3">
            <div className="metric-mini"><span>Track ID</span><strong>{track?.track_id}</strong></div>
            <div className="metric-mini"><span>Valid time / lead</span><strong>{selectedPoint.valid_time.replace("T", " ")} · +{selectedPoint.lead_time_hours ?? "?"}h</strong></div>
            <div className="metric-mini"><span>Latitude / longitude</span><strong>{selectedPoint.latitude.toFixed(3)}°, {selectedPoint.longitude.toFixed(3)}°</strong></div>
            <div className="metric-mini"><span>EFI-like / GNN score</span><strong>{selectedPoint.efi?.toFixed(3) ?? "N/A"} / {selectedPoint.confidence?.toFixed(3) ?? "N/A"}</strong></div>
            <div className="metric-mini col-span-2"><span>Geographic bounds</span><strong>{formatBounds(selectedPoint.geographic_bbox)}</strong></div>
          </div> : <p className="mt-3 text-sm text-muted-foreground">No thresholded anomaly track at this forecast time.</p>}
          <p className="mt-3 text-xs text-muted-foreground">Track is an experimental anomaly cluster, not a verified cyclone-center track. {replay.model_status.classification}; no accuracy claimed.</p>
        </section>
      </div>
      <section className="panel mt-4 p-4">
        <div className="data-label">Pipeline provenance and scientific limitations</div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {[
            ["Forecast", "Real NCMRWF TIGGE perturbed ensemble; 11 members"],
            ["Reference", "IMDAA May 2011–2019 t2m only; excludes 2020"],
            ["Wind anomaly", "Within-forecast spatial percentile proxy; not climatological EFI"],
            ["GNN / tracking", "Weakly supervised, uncalibrated experimental output"],
            ["Rainfall", "Accumulated tp; excluded from anomaly analysis; not a rate"],
            ["Physics", replay.physics.physics_status],
            ["Independent verification", replay.verification.verification_status],
            ["Downscaling", "Not implemented or validated in this replay"],
          ].map(([label, value]) => <div className="metric-mini" key={label}><span>{label}</span><strong>{value}</strong></div>)}
        </div>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted-foreground">{replay.data_limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
      </section>
    </>}
  </>;
}

function FieldRaster({
  field,
  minimum,
  maximum,
  palette,
}: {
  field: AmphanField;
  minimum: number;
  maximum: number;
  palette: "temperature" | "efi";
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const renderStartedAt = performance.now();
    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(bounds.width * pixelRatio));
    canvas.height = Math.max(1, Math.round(bounds.height * pixelRatio));
    context.setTransform(canvas.width / 600, 0, 0, canvas.height / 470, 0, 0);
    context.clearRect(0, 0, 600, 470);
    const latitudeMinimum = Math.min(...field.latitude);
    const latitudeMaximum = Math.max(...field.latitude);
    const longitudeMinimum = Math.min(...field.longitude);
    const longitudeMaximum = Math.max(...field.longitude);
    const stride = Math.max(1, Math.ceil(Math.max(field.latitude.length, field.longitude.length) / 64));
    const project = (latitude: number, longitude: number) => ({
      x: longitudeMaximum === longitudeMinimum ? 300 : 10 + ((longitude - longitudeMinimum) / (longitudeMaximum - longitudeMinimum)) * 580,
      y: latitudeMaximum === latitudeMinimum ? 230 : 450 - ((latitude - latitudeMinimum) / (latitudeMaximum - latitudeMinimum)) * 440,
    });
    const edge = (coordinates: number[], index: number) => {
      const current = coordinates[index];
      if (current === undefined) return [Number.NaN, Number.NaN] as const;
      const previous = coordinates[index - 1];
      const next = coordinates[index + 1];
      const first = previous === undefined ? current : (previous + current) / 2;
      const second = next === undefined ? current : (current + next) / 2;
      const minimumCoordinate = Math.min(coordinates[0] ?? current, coordinates[coordinates.length - 1] ?? current);
      const maximumCoordinate = Math.max(coordinates[0] ?? current, coordinates[coordinates.length - 1] ?? current);
      return [
        Math.max(minimumCoordinate, Math.min(first, second)),
        Math.min(maximumCoordinate, Math.max(first, second)),
      ] as const;
    };
    for (let row = 0; row < field.latitude.length; row += stride) {
      const latitude = field.latitude[row];
      if (latitude === undefined) continue;
      const [south, north] = edge(field.latitude, row);
      for (let column = 0; column < field.longitude.length; column += stride) {
        const value = field.values[row]?.[column];
        const longitude = field.longitude[column];
        if (value === undefined || longitude === undefined || !Number.isFinite(value)) continue;
        const normalized = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum);
        const [west, east] = edge(field.longitude, column);
        const upperLeft = project(north, west);
        const lowerRight = project(south, east);
        const hue = palette === "efi"
          ? Math.round(220 - Math.max(0, normalized) * 220)
          : Math.round(220 - normalized * 220);
        const lightness = palette === "efi"
          ? Math.round(25 + normalized * 35)
          : Math.round(30 + normalized * 35);
        const saturation = palette === "efi" ? 85 : 80;
        context.fillStyle = `hsl(${hue} ${saturation}% ${lightness}%)`;
        context.fillRect(
          upperLeft.x,
          upperLeft.y,
          Math.max(0.5, lowerRight.x - upperLeft.x + 0.2),
          Math.max(0.5, lowerRight.y - upperLeft.y + 0.2),
        );
      }
    }
    if (import.meta.env["DEV"]) {
      console.debug("[STORMTRACE_MAP_RENDER]", {
        valid_time: field.valid_time,
        variable: field.variable,
        elapsed_ms: Number((performance.now() - renderStartedAt).toFixed(1)),
        cells_drawn: Math.ceil(field.latitude.length / stride) * Math.ceil(field.longitude.length / stride),
      });
    }
  }, [field, maximum, minimum, palette]);
  return <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 size-full" />;
}

function GeographicField({
  field,
  title,
  trackPoints,
  center,
  bounds,
  palette,
  geographicSource = "IMDAA",
}: {
  field: AmphanField | undefined;
  title: string;
  trackPoints: AmphanTrackPoint[];
  center: { latitude: number; longitude: number } | undefined;
  bounds: AmphanGeographicBounds | undefined;
  palette: "temperature" | "efi";
  geographicSource?: string;
}) {
  if (!field || field.values.length === 0 || field.latitude.length === 0 || field.longitude.length === 0) {
    return <section className="panel p-4"><div className="data-label">{title}</div><div className="mt-6 text-sm text-muted-foreground">Field unavailable.</div></section>;
  }
  const latitudeMinimum = Math.min(...field.latitude);
  const latitudeMaximum = Math.max(...field.latitude);
  const longitudeMinimum = Math.min(...field.longitude);
  const longitudeMaximum = Math.max(...field.longitude);
  if (![latitudeMinimum, latitudeMaximum, longitudeMinimum, longitudeMaximum].every(Number.isFinite)) {
    return <section className="panel p-4"><div className="data-label">{title}</div><div className="mt-6 text-sm text-muted-foreground">Field coordinates unavailable.</div></section>;
  }
  const values = field.values.flat();
  const minimum = values.reduce((current, value) => Math.min(current, value), Number.POSITIVE_INFINITY);
  const maximum = values.reduce((current, value) => Math.max(current, value), Number.NEGATIVE_INFINITY);
  const position = (latitude: number, longitude: number) => ({
    x: longitudeMaximum === longitudeMinimum ? 300 : 10 + ((longitude - longitudeMinimum) / (longitudeMaximum - longitudeMinimum)) * 580,
    y: latitudeMaximum === latitudeMinimum ? 230 : 450 - ((latitude - latitudeMinimum) / (latitudeMaximum - latitudeMinimum)) * 440,
  });
  const centerPosition = center
    && center.latitude >= latitudeMinimum && center.latitude <= latitudeMaximum
    && center.longitude >= longitudeMinimum && center.longitude <= longitudeMaximum
    ? position(center.latitude, center.longitude)
    : undefined;
  const points = Array.isArray(trackPoints)
    ? trackPoints.filter((point) =>
      Number.isFinite(point.latitude) && Number.isFinite(point.longitude)
      && point.latitude >= latitudeMinimum && point.latitude <= latitudeMaximum
      && point.longitude >= longitudeMinimum && point.longitude <= longitudeMaximum,
    )
    : [];
  const trackLine = points.map((point) => {
    const projected = position(point.latitude, point.longitude);
    return `${projected.x},${projected.y}`;
  }).join(" ");
  return <section className="panel p-4">
    <div className="flex items-center justify-between"><div className="data-label">{title}</div><span className="text-[10px] text-muted-foreground">{field.valid_time ?? ""} {field.units ?? ""}</span></div>
    <div className="relative mt-3 h-[420px] w-full overflow-hidden rounded bg-slate-950 md:h-[470px]">
    <FieldRaster field={field} minimum={minimum} maximum={maximum} palette={palette} />
    <svg viewBox="0 0 600 470" role="img" aria-label={`${title} geographic map`} className="absolute inset-0 block size-full">
      {Array.from({ length: 5 }, (_, index) => {
        const longitude = longitudeMinimum + (longitudeMaximum - longitudeMinimum) * index / 4;
        const x = position(latitudeMinimum, longitude).x;
        const latitude = latitudeMinimum + (latitudeMaximum - latitudeMinimum) * index / 4;
        const y = position(latitude, longitudeMinimum).y;
        return <g key={`geo-grid-${index}`} aria-hidden="true">
          <line x1={x} y1="10" x2={x} y2="450" stroke="#94a3b8" strokeOpacity=".24" strokeDasharray="3 5" />
          <text x={x} y="458" fill="#cbd5e1" fontSize="10" textAnchor="middle">{longitude.toFixed(1)}°E</text>
          <line x1="10" y1={y} x2="590" y2={y} stroke="#94a3b8" strokeOpacity=".24" strokeDasharray="3 5" />
          <text x="8" y={y - 3} fill="#cbd5e1" fontSize="10" textAnchor="start">{latitude.toFixed(1)}°N</text>
        </g>;
      })}
      {trackLine && <polyline points={trackLine} fill="none" stroke="#ffffff" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />}
      {bounds && <GeoBounds bounds={bounds} position={position} />}
      {centerPosition && <circle cx={centerPosition.x} cy={centerPosition.y} r="6" fill="#f43f5e" stroke="white" strokeWidth="2" />}
    </svg>
    </div>
    <div className="mt-2 flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
      <span className="flex min-w-40 flex-1 items-center gap-2">
        <span>{minimum.toFixed(1)} {field.units ?? ""}</span>
        <i className="h-2 flex-1 rounded bg-gradient-to-r from-blue-600 via-yellow-400 to-red-600" />
        <span>{maximum.toFixed(1)} {field.units ?? ""}</span>
      </span>
      <span className="flex items-center gap-1"><i className="size-2 rounded-full bg-rose-500" />EFI alert maximum</span>
      <span className="flex items-center gap-1"><i className="h-0.5 w-4 bg-white" />experimental GNN cluster-centroid path</span>
      {bounds && <span className="flex items-center gap-1"><i className="size-3 border border-amber-300" />active GNN cluster bounds</span>}
      <span>Geographic coordinates from {geographicSource} · visualization sampled to {field.returned_shape?.join("×") ?? "source grid"}</span>
    </div>
  </section>;
}

function ReplaySummary({ replay, alert, trackPoint, trackId }: { replay: AmphanOverview; alert: AmphanAlert | undefined; trackPoint: AmphanTrackPoint | undefined; trackId: string | undefined }) {
  return <section className="panel p-4">
    <div className="data-label">STORMTRACE ALERT</div>
    <div className="mt-3 flex items-start justify-between gap-3"><div><div className="text-lg font-semibold">Cyclone Amphan</div><div className="text-xs text-muted-foreground">{alert?.time.replace("T", " ") ?? replay.event_period}</div></div><div className="data-mode">{alert?.status ?? replay.mode}</div></div>
    <div className="mt-4 grid grid-cols-2 gap-3">
      <div className="metric-mini"><span>Location latitude</span><strong>{alert?.location.latitude.toFixed(3) ?? "N/A"}°</strong></div>
      <div className="metric-mini"><span>Location longitude</span><strong>{alert?.location.longitude.toFixed(3) ?? "N/A"}°</strong></div>
      <div className="metric-mini"><span>Prototype severity</span><strong>{alert?.severity ?? replay.alert.severity}</strong></div>
      <div className="metric-mini"><span>EFI</span><strong>{alert?.efi.toFixed(3) ?? "N/A"}</strong></div>
      <div className="metric-mini"><span>Confidence</span><strong>{alert ? `${(alert.confidence * 100).toFixed(1)}% · uncalibrated` : "N/A"}</strong></div>
      <div className="metric-mini"><span>Track ID</span><strong>{trackPoint ? trackId ?? "Available" : "No track at time"}</strong></div>
    </div>
    <p className="mt-4 rounded border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-200">Historical replay only. Not a live operational warning.</p>
  </section>;
}

function DownscaledPreview({ title, field }: { title: string; field: AmphanField | undefined }) {
  if (!field || !field.values.length) return <div className="metric-mini"><span>{title}</span><strong>Unavailable</strong></div>;
  const flattened = field.values.flat();
  const min = flattened.reduce((value, current) => Math.min(value, current), Number.POSITIVE_INFINITY);
  const max = flattened.reduce((value, current) => Math.max(value, current), Number.NEGATIVE_INFINITY);
  const stride = Math.max(1, Math.ceil(Math.max(field.values.length, field.values[0]?.length ?? 0) / 30));
  const statistics = field.statistics;
  return <div className="metric-mini"><span>{title}</span><strong>{field.classification}</strong><span>min {statistics?.minimum.toFixed(2) ?? min.toFixed(2)} · max {statistics?.maximum.toFixed(2) ?? max.toFixed(2)} K</span><span>mean {statistics?.mean.toFixed(2) ?? "N/A"} · p05/p50/p95 {statistics ? `${statistics.p05.toFixed(2)} / ${statistics.p50.toFixed(2)} / ${statistics.p95.toFixed(2)} K` : "N/A"}</span><div className="mt-2 grid" style={{ gridTemplateColumns: `repeat(${Math.ceil((field.values[0]?.length ?? 1) / stride)}, minmax(0, 1fr))` }}>
    {field.values.filter((_, index) => index % stride === 0).flatMap((row, rowIndex) => row.filter((_, index) => index % stride === 0).map((value, index) => <span key={`${rowIndex}-${index}`} className="aspect-square" style={{ backgroundColor: `hsl(${Math.round(220 - ((value - min) / Math.max(1e-6, max - min)) * 220)} 75% 45%)` }} />))}
  </div></div>;
}

function ReplayPipelineCard({ stages }: { stages: string[] }) {
  return <section className="panel mb-4 p-4">
    <div className="data-label">Historical replay pipeline · cached outputs</div>
    <div className="mt-3 flex flex-wrap items-center gap-2">
      {stages.map((stage, index) => <Fragment key={stage}>
        <span className={`metric-mini ${stage.includes("experimental") ? "border-amber-500/30 text-amber-200" : ""}`}>
          {stage.includes("experimental") ? `${stage} · EXPERIMENTAL` : stage}
        </span>
        {index < stages.length - 1 && <span aria-hidden="true" className="text-muted-foreground">↓</span>}
      </Fragment>)}
    </div>
    <p className="mt-3 text-[10px] text-muted-foreground">Replay reads precomputed results. It does not run model training in the browser request.</p>
  </section>;
}

function ModelComparison({
  coarse,
  unet,
  diffusion,
}: {
  coarse: AmphanField | undefined;
  unet: AmphanField | undefined;
  diffusion: AmphanField | undefined;
}) {
  return <section className="panel mt-4 p-4">
    <div className="data-label">Experimental amplitude-preserving reconstruction</div>
    <p className="mt-1 text-xs text-muted-foreground">Cached field previews and descriptive statistics only; no accuracy or verification claim.</p>
    <div className="mt-3 grid gap-3 lg:grid-cols-3">
      <DownscaledPreview title="Input / coarse IMDAA crop" field={coarse} />
      <DownscaledPreview title="U-Net · EXPERIMENTAL" field={unet} />
      <DownscaledPreview title="Diffusion · EXPERIMENTAL" field={diffusion} />
    </div>
    <p className="mt-3 text-xs text-amber-200">Independent high-resolution verification unavailable.</p>
  </section>;
}

function DataIntegrityPanel({ replay }: { replay: AmphanOverview }) {
  const summary = replay.demo_summary;
  const bounds = summary.spatial_bounds;
  return <section className="panel p-4">
    <div className="data-label">Data integrity & limitations</div>
    <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
      <IntegrityItem label="Dataset" value={summary.dataset} />
      <IntegrityItem label="Forecast source" value="NEPS-G not connected" />
      <IntegrityItem label="Reference" value={summary.reference_period} />
      <IntegrityItem label="Event holdout" value="May 16–21, 2020" />
      <IntegrityItem label="Grid" value={summary.grid_dimensions ? `${summary.grid_dimensions.latitude} × ${summary.grid_dimensions.longitude}` : "Unavailable"} />
      <IntegrityItem label="Spatial bounds" value={bounds ? `${bounds.latitude_min.toFixed(2)}–${bounds.latitude_max.toFixed(2)}°N, ${bounds.longitude_min.toFixed(2)}–${bounds.longitude_max.toFixed(2)}°E` : "Unavailable"} />
      <IntegrityItem label="Rainfall" value="Raw APCP preserved; semantics unresolved; excluded from analysis" />
      <IntegrityItem label="Specific humidity q" value="Unavailable" />
      <IntegrityItem label="Physics" value={replay.physics.physics_status} />
      <IntegrityItem label="Independent verification" value={replay.verification.verification_status} />
    </dl>
    <ul className="mt-3 list-disc space-y-1 pl-4 text-[10px] text-muted-foreground">
      {summary.data_limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
    </ul>
  </section>;
}

function IntegrityItem({ label, value }: { label: string; value: string }) {
  return <div className="metric-mini"><dt>{label}</dt><dd className="mt-1 font-semibold">{value}</dd></div>;
}

function formatBounds(bounds: AmphanGeographicBounds) {
  return `${bounds.latitude_min.toFixed(2)}–${bounds.latitude_max.toFixed(2)}°N, ${bounds.longitude_west.toFixed(2)}–${bounds.longitude_east.toFixed(2)}°E`;
}

function GeoBounds({
  bounds,
  position,
}: {
  bounds: AmphanGeographicBounds;
  position: (latitude: number, longitude: number) => { x: number; y: number };
}) {
  const northwest = position(bounds.latitude_max, bounds.longitude_west);
  const southeast = position(bounds.latitude_min, bounds.longitude_east);
  const x = Math.max(10, Math.min(northwest.x, southeast.x));
  const y = Math.max(10, Math.min(northwest.y, southeast.y));
  const width = Math.max(0, Math.min(590, Math.max(northwest.x, southeast.x)) - x);
  const height = Math.max(0, Math.min(450, Math.max(northwest.y, southeast.y)) - y);
  return <rect x={x} y={y} width={width} height={height} fill="none" stroke="#fcd34d" strokeWidth="2" strokeDasharray="5 4" />;
}

export function TrackingPage() {
  const queryClient = useQueryClient();
  const [timeIndex, setTimeIndex] = useState(0);
  const [layer, setLayer] = useState<"t2m" | "anomaly_score" | "gnn_probability" | "rainfall" | "track">("anomaly_score");
  const [playing, setPlaying] = useState(false);
  const health = useBackendHealth();
  const overviewQuery = useQuery({
    queryKey: ["tigge-amphan-overview"],
    queryFn: ({ signal }) => api.tiggeAmphanOverview(signal),
    staleTime: Infinity,
    retry: false,
  });
  const replayMutation = useMutation({
    mutationFn: ({ signal }: { signal?: AbortSignal }) => api.runTiggeAmphanReplay(signal),
    onSuccess: (overview) => {
      queryClient.setQueryData(["tigge-amphan-overview"], overview);
      setTimeIndex(0);
      setPlaying(false);
    },
    onError: (error) => toast.error("TIGGE REPLAY UNAVAILABLE", { description: error.message }),
  });
  const replay = replayMutation.data ?? overviewQuery.data;
  const selectedVariable = layer === "track" ? "anomaly_score" : layer;
  const fieldQuery = useQuery({
    queryKey: ["tigge-amphan-field", selectedVariable, timeIndex],
    queryFn: ({ signal }) => api.tiggeAmphanField(selectedVariable, timeIndex, signal, true),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const alertQuery = useQuery({
    queryKey: ["tigge-amphan-alert", timeIndex],
    queryFn: ({ signal }) => api.tiggeAmphanAlert(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const tracksQuery = useQuery({
    queryKey: ["tigge-amphan-tracks", timeIndex],
    queryFn: ({ signal }) => api.tiggeAmphanTracks(timeIndex, signal),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const timeline = replay?.timeline ?? [];
  const selectedTime = timeline[timeIndex];
  useEffect(() => {
    if (!playing || timeline.length < 2) return;
    const timer = window.setInterval(() => {
      setTimeIndex((current) => current >= timeline.length - 1 ? 0 : current + 1);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [playing, timeline.length]);

  const fallbackTrack = replay?.dominant_track;
  const activeTrack = tracksQuery.data?.dominant_track ?? fallbackTrack;
  const allTrackPoints = activeTrack?.track_points ?? [];
  const selectedTrackPoint = tracksQuery.data?.dominant_track?.selected_track_point
    ?? allTrackPoints.find((point) => point.valid_time === selectedTime?.valid_time);
  const previousTrackPoints = allTrackPoints.filter((point) =>
    point.valid_time <= (selectedTime?.valid_time ?? ""),
  );
  const displayedField = fieldQuery.data;
  const isLoading = overviewQuery.isPending && !replayMutation.data;
  const isUnavailable = health.isError || overviewQuery.isError || replayMutation.isError;
  const layers = [
    ["t2m", "Temperature · t2m"],
    ["anomaly_score", "EFI-like / anomaly proxy"],
    ["gnn_probability", "GNN probability"],
    ["rainfall", "Accumulated precipitation · tp"],
    ["track", "AI-derived anomaly track"],
  ] as const;
  return <>
    <PageHeading
      eyebrow="REAL NCMRWF TIGGE · EXPERIMENTAL AI"
      title="Anomaly Tracking"
      description="Spatio-Temporal Extreme Weather Detection & Tracking · cached Cyclone Amphan forecast replay."
      action={<Button onClick={() => replayMutation.mutate({})} disabled={replayMutation.isPending}>
        {replayMutation.isPending ? <LoaderCircle className="animate-spin" /> : <Play />}
        {replay ? "Reload cached replay" : "Load Amphan replay"}
      </Button>}
    />
    <div className="mb-4 flex flex-wrap gap-2">
      <span className="data-mode">REAL NCMRWF TIGGE</span>
      <span className="data-mode">{String((replay?.provenance["forecast"] as Record<string, unknown> | undefined)?.["ensemble_count"] ?? "SOURCE ENSEMBLE")} MEMBERS</span>
      <span className="data-mode border-amber-500/30 text-amber-200">EXPERIMENTAL AI</span>
    </div>
    {isLoading && <section className="panel p-5 text-sm text-muted-foreground">Loading cached TIGGE replay metadata…</section>}
    {isUnavailable && <section role="alert" className="panel mb-4 border border-destructive/40 p-4 text-sm">
      <div className="font-semibold text-destructive">{health.isError ? "BACKEND OFFLINE" : "TIGGE REPLAY UNAVAILABLE"}</div>
      <p className="mt-2 text-muted-foreground">
        {health.isError
          ? "No replay data is cached in this browser. Start the backend with the prepared TIGGE cache, then reload this page."
          : replayMutation.error instanceof Error ? replayMutation.error.message : "The cached replay metadata could not be loaded."}
      </p>
      <Button className="mt-3" variant="outline" onClick={() => {
        void overviewQuery.refetch();
        void health.refetch();
      }}>Retry</Button>
    </section>}
    {replay && <>
      <section className="panel mb-4 p-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <IntegrityItem label="Event" value={replay.event} />
          <IntegrityItem label="Initialization" value={`${replay.forecast_initialization_time.replace("T", " ")} UTC`} />
          <IntegrityItem label="Forecast range" value={timeline.length > 0 ? `${timeline[0]?.lead_time_hours ?? "?"}–${timeline[timeline.length - 1]?.lead_time_hours ?? "?"} h · ${timeline.length} valid times` : "Unavailable"} />
          <IntegrityItem label="Ensemble / grid" value={`${String((replay.provenance["forecast"] as Record<string, unknown>)["ensemble_count"] ?? "Unavailable")} source members · ${String((replay.provenance["forecast"] as Record<string, unknown>)["grid_dimensions"] ?? "Unavailable")} source; ${replay.grid_dimensions.latitude} × ${replay.grid_dimensions.longitude} analysis`} />
          <IntegrityItem label="Reference" value="IMDAA May 2011–2019 t2m" />
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          <IntegrityItem label="Temperature anomaly" value="EXPERIMENTAL EFI-like" />
          <IntegrityItem label="Wind anomaly" value="PROXY · within-forecast rank" />
          <IntegrityItem label="GNN / tracking" value="EXPERIMENTAL / DERIVED" />
          <IntegrityItem label="Verification" value="NOT AVAILABLE" />
        </div>
      </section>
      <section className="panel mb-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="data-label">Forecast timeline · {timeline.length} cached valid times</div>
            <div className="mt-1 font-mono text-sm">{selectedTime?.valid_time ?? "N/A"} · +{selectedTime?.lead_time_hours ?? "N/A"} h</div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" aria-label="Previous forecast step" onClick={() => { setPlaying(false); setTimeIndex((current) => Math.max(0, current - 1)); }} disabled={timeIndex === 0}><SkipBack /></Button>
            <Button variant="outline" size="icon" aria-label={playing ? "Pause forecast timeline" : "Play forecast timeline"} onClick={() => setPlaying((value) => !value)} disabled={timeline.length < 2}>{playing ? <Pause /> : <Play />}</Button>
            <Button variant="outline" size="icon" aria-label="Next forecast step" onClick={() => { setPlaying(false); setTimeIndex((current) => Math.min(timeline.length - 1, current + 1)); }} disabled={timeIndex >= timeline.length - 1}><SkipForward /></Button>
          </div>
        </div>
        <input
          aria-label="Forecast time"
          type="range"
          min={0}
          max={Math.max(0, timeline.length - 1)}
          value={timeIndex}
          onChange={(event) => { setPlaying(false); setTimeIndex(Number(event.target.value)); }}
          className="mt-4 w-full accent-primary"
        />
        <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
        <div className="flex max-w-full flex-nowrap gap-2 overflow-x-auto pb-2">
          {timeline.map((item, index) => <button key={item.valid_time} className={`metric-mini shrink-0 text-left ${index === timeIndex ? "border-primary bg-primary/10" : ""}`} onClick={() => { setPlaying(false); setTimeIndex(index); }}>
            <span>+{String(item.lead_time_hours).padStart(2, "0")}h</span><strong>{item.valid_time.slice(5, 16).replace("T", " ")}</strong>
          </button>)}
        </div>
          <label className="flex items-center gap-2 text-xs">
            <span className="data-label">Map layer</span>
            <select className="rounded border border-border bg-card px-3 py-2" value={layer} onChange={(event) => setLayer(event.target.value as typeof layer)}>
              {layers.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
        </div>
      </section>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(280px,.7fr)]">
        <div>
          {fieldQuery.isError
            ? <section role="alert" className="panel border border-destructive/40 p-4 text-sm">Selected geographic field is unavailable: {fieldQuery.error.message}</section>
            : <GeographicField
              field={displayedField}
              title={`${layers.find(([value]) => value === layer)?.[1] ?? layer}${layer === "rainfall" ? " · accumulated amount, not a rate" : ""}`}
              trackPoints={layer === "track" ? previousTrackPoints : []}
              center={layer === "track" && selectedTrackPoint ? { latitude: selectedTrackPoint.latitude, longitude: selectedTrackPoint.longitude } : undefined}
              bounds={layer === "track" ? selectedTrackPoint?.geographic_bbox : undefined}
              palette={layer === "t2m" || layer === "rainfall" ? "temperature" : "efi"}
              geographicSource="NCMRWF TIGGE"
            />}
          <p className="mt-2 text-[10px] text-muted-foreground">
            Coordinates are the actual TIGGE latitude/longitude grid. {displayedField?.visualization_downsampled ? `Visualization is sampled from ${displayedField.source_shape?.join("×") ?? "the source grid"} to ${displayedField.returned_shape?.join("×") ?? "a display grid"}; this does not alter cached analysis.` : ""}
          </p>
        </div>
        <section className="panel border border-amber-500/30 p-4">
          <div className="data-label">Project-specific alert · not an official warning</div>
          {alertQuery.isError
            ? <p role="alert" className="mt-3 text-sm text-destructive">Alert unavailable: {alertQuery.error.message}</p>
            : alertQuery.data ? <>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <IntegrityItem label="Alert level" value={alertQuery.data.severity} />
                <IntegrityItem label="Severity" value="Project-specific heuristic" />
                <IntegrityItem label="Center latitude" value={`${alertQuery.data.location.latitude.toFixed(3)}°`} />
                <IntegrityItem label="Center longitude" value={`${alertQuery.data.location.longitude.toFixed(3)}°`} />
                <IntegrityItem label="Anomaly score" value={alertQuery.data.anomaly_score.toFixed(3)} />
                <IntegrityItem label="GNN probability" value={`${(alertQuery.data.confidence * 100).toFixed(1)}% · uncalibrated`} />
                <IntegrityItem label="Lead time" value={`+${alertQuery.data.lead_time_hours} h`} />
                <IntegrityItem label="Source / model" value={`${alertQuery.data.source} · ${alertQuery.data.model_version}`} />
              </div>
              <p className="mt-3 rounded border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-100">{alertQuery.data.notice}</p>
            </> : <p className="mt-3 text-sm text-muted-foreground">Loading alert…</p>}
          <div className="mt-4 border-t border-border pt-4">
            <div className="data-label">Ensemble spread</div>
            <p className="mt-2 text-xs text-muted-foreground">
              The forecast source contains 11 perturbed members, but this replay cache retains ensemble-mean analysis fields only. Per-member peaks, centroids, and spread are unavailable; no spread interval is inferred.
            </p>
          </div>
        </section>
      </div>
      <section className="panel mt-4 p-4">
        <div className="data-label">Spatio-temporal anomaly evolution · unsmoothed cached maxima</div>
        <div className="mt-3 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeline} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="lead_time_hours" tickFormatter={(value: number) => `+${value}h`} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
              <ChartTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }} labelFormatter={(value) => `Lead +${value}h`} />
              <Legend />
              {selectedTime && <ReferenceLine x={selectedTime.lead_time_hours} stroke="var(--foreground)" strokeDasharray="4 4" />}
              <Line type="linear" dataKey="anomaly_max" name="Anomaly score max" stroke="#f59e0b" dot={false} isAnimationActive={false} />
              <Line type="linear" dataKey="gnn_probability_max" name="GNN probability max" stroke="#22d3ee" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
      <section className="panel mt-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div><div className="data-label">AI-derived anomaly track · not an official cyclone track</div><div className="mt-1 text-xs text-muted-foreground">Geographic detections retained in the prepared replay.</div></div>
          <span className="data-mode">{allTrackPoints.length} points in selected dominant track</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-xs">
            <thead><tr className="border-b border-border text-muted-foreground"><th className="p-2">Valid time</th><th className="p-2">Latitude</th><th className="p-2">Longitude</th><th className="p-2">Severity</th><th className="p-2">Anomaly score</th><th className="p-2">GNN probability</th><th className="p-2">Coverage</th></tr></thead>
            <tbody>{allTrackPoints.map((point) => {
              const step = timeline.find((item) => item.valid_time === point.valid_time);
              return <tr key={`${point.valid_time}-${point.latitude}-${point.longitude}`} className={`border-b border-border/60 ${point.valid_time === selectedTime?.valid_time ? "bg-primary/10" : ""}`}>
                <td className="p-2">{point.valid_time.replace("T", " ")} · +{point.lead_time_hours ?? "N/A"}h</td>
                <td className="p-2">{point.latitude.toFixed(3)}°</td>
                <td className="p-2">{point.longitude.toFixed(3)}°</td>
                <td className="p-2">{point.severity ?? "N/A"}</td>
                <td className="p-2">{point.anomaly_score?.toFixed(3) ?? "N/A"}</td>
                <td className="p-2">{point.confidence?.toFixed(3) ?? "N/A"}</td>
                <td className="p-2">{step?.anomaly_coverage === undefined ? "N/A" : `${(step.anomaly_coverage * 100).toFixed(2)}%`}</td>
              </tr>;
            })}</tbody>
          </table>
          {!allTrackPoints.length && <p className="py-6 text-center text-sm text-muted-foreground">No derived track points are available in the cached replay.</p>}
        </div>
      </section>
      <details className="panel mt-4 p-4">
        <summary className="cursor-pointer text-sm font-semibold">How it works · research prototype</summary>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-xs text-muted-foreground">
          <li>NCMRWF TIGGE supplies the real 11-member forecast source; cached fields are ensemble means.</li>
          <li>Forecast temperature is compared with the UTC-hour-matched May 2011–2019 IMDAA t2m reference, excluding 2020.</li>
          <li>Temperature EFI-like scores and within-forecast wind spatial-percentile proxies mark candidate anomalies.</li>
          <li>An experimental, weakly supervised geographic GNN assigns uncalibrated per-grid scores.</li>
          <li>Temporal association links thresholded geographic detections into derived tracks.</li>
          <li>Project-specific severity and alert summaries are research outputs, not operational meteorological warnings.</li>
        </ol>
      </details>
    </>}
  </>;
}

export function DownscalingPage() {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState<"unet" | "diffusion">("unet");
  const [transectLatitude, setTransectLatitude] = useState<number | null>(null);
  const [forecastIndex, setForecastIndex] = useState(0);
  const [forecastVariable, setForecastVariable] = useState<"t2m" | "rainfall">("t2m");
  const health = useBackendHealth();
  const tiggeQuery = useQuery({
    queryKey: ["tigge-amphan-overview"],
    queryFn: ({ signal }) => api.tiggeAmphanOverview(signal),
    staleTime: Infinity,
    retry: false,
  });
  const overviewQuery = useQuery({
    queryKey: ["amphan-overview"],
    queryFn: ({ signal }) => api.amphanOverview(signal),
    staleTime: Infinity,
    retry: false,
  });
  const refreshCache = useMutation({
    mutationFn: ({ signal }: { signal?: AbortSignal }) => api.runAmphanReplay(signal),
    onSuccess: (overview) => {
      queryClient.setQueryData(["amphan-overview"], overview);
      void queryClient.invalidateQueries({ queryKey: ["amphan-field"] });
    },
    onError: (error) => toast.error("CACHED DEMONSTRATION UNAVAILABLE", { description: error.message }),
  });
  const replay = refreshCache.data ?? overviewQuery.data;
  const inputQuery = useQuery({
    queryKey: ["amphan-field", "crop_t2m"],
    queryFn: ({ signal }) => api.amphanField("crop_t2m", 0, signal, true),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const unetQuery = useQuery({
    queryKey: ["amphan-field", "unet"],
    queryFn: ({ signal }) => api.amphanField("unet", 0, signal, true),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const diffusionQuery = useQuery({
    queryKey: ["amphan-field", "diffusion"],
    queryFn: ({ signal }) => api.amphanField("diffusion", 0, signal, true),
    enabled: Boolean(replay),
    staleTime: Infinity,
    retry: false,
  });
  const tigge = tiggeQuery.data;
  const tiggeFieldQuery = useQuery({
    queryKey: ["tigge-amphan-field", forecastVariable, forecastIndex],
    queryFn: ({ signal }) => api.tiggeAmphanField(forecastVariable, forecastIndex, signal, true, 32),
    enabled: Boolean(tigge),
    staleTime: Infinity,
    retry: false,
  });
  const tiggeTemperatureQuery = useQuery({
    queryKey: ["tigge-amphan-field", "t2m", forecastIndex],
    queryFn: ({ signal }) => api.tiggeAmphanField("t2m", forecastIndex, signal, true, 32),
    enabled: Boolean(tigge),
    staleTime: Infinity,
    retry: false,
  });
  const input = inputQuery.data;
  const unet = unetQuery.data;
  const diffusion = diffusionQuery.data;
  const peakIndex = replay?.timeline.reduce(
    (best, item, index, items) => item.efi_max > (items[best]?.efi_max ?? Number.NEGATIVE_INFINITY) ? index : best,
    0,
  ) ?? 0;
  const snapshotTime = replay?.timeline[peakIndex]?.time ?? "N/A";
  const forecastStep = tigge?.timeline[forecastIndex];
  const profile = makeTransectProfile(tiggeTemperatureQuery.data, unet, diffusion, transectLatitude);
  const transectValues = profile.flatMap((point) =>
    [point.tiggeInput, point.unet, point.diffusion].filter(
      (value): value is number => typeof value === "number" && Number.isFinite(value),
    ),
  );
  const transectMinimum = transectValues.length ? Math.min(...transectValues) : 0;
  const transectMaximum = transectValues.length ? Math.max(...transectValues) : 1;
  const transectPadding = Math.max((transectMaximum - transectMinimum) * 0.1, 0.2);
  const transectDomain: [number, number] = [
    Math.floor((transectMinimum - transectPadding) * 10) / 10,
    Math.ceil((transectMaximum + transectPadding) * 10) / 10,
  ];
  const commonLatitudes = tiggeTemperatureQuery.data?.latitude ?? [];
  const suggestedTransect = commonLatitudes.length ? commonLatitudes[Math.floor(commonLatitudes.length / 2)] : null;
  const currentTransectLatitude = transectLatitude ?? suggestedTransect;
  const loading = overviewQuery.isPending;
  const error = health.isError
    ? "Backend unavailable. No cached replay data has been stored in this browser."
    : overviewQuery.isError
      ? overviewQuery.error.message
      : [inputQuery, unetQuery, diffusionQuery].find((query) => query.isError)?.error?.message;
  const stats = [input, unet, diffusion].map((field) => field?.statistics);
  const peakValues = stats.map((item) => item?.maximum);
  const tailAmplitudes = stats.map((item) => item ? item.p95 - item.p50 : undefined);
  const inputTail = tailAmplitudes[0];
  const tailRatios = tailAmplitudes.map((value) =>
    value === undefined || inputTail === undefined || Math.abs(inputTail) < 1e-9 ? undefined : value / inputTail,
  );
  const modelFields = [
    { variable: "crop_t2m", title: "Input · cached IMDAA event crop", field: input, palette: "temperature" as const },
    { variable: "unet", title: "U-Net baseline · experimental", field: unet, palette: "temperature" as const },
    { variable: "diffusion", title: "Conditional diffusion · experimental", field: diffusion, palette: "temperature" as const },
  ];
  return <>
    <PageHeading
      eyebrow="Real forecast context · separate cached experimental snapshots"
      title="Downscaling Lab"
      description="A real NCMRWF TIGGE input view alongside separate, cached experimental model outputs. No TIGGE-matched downscaling is available."
      action={<span className="data-mode border-amber-500/30 text-amber-200">EXPERIMENTAL</span>}
    />
    <section className="panel mb-4 p-4">
      <p className="text-sm text-muted-foreground">
        The TIGGE field below is real forecast input. Existing U-Net and diffusion outputs are cached experimental snapshots from an IMDAA historical crop; they were not generated from or matched to the selected TIGGE timestep. The sequence below is a project workflow illustration, not a claim that these models consumed this TIGGE field.
      </p>
      <div className="mt-4">
        <div className="data-label">Pipeline and status · historical cached demonstration</div>
        <div className="mt-3 grid items-stretch gap-2 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          <div className="metric-mini"><span>INPUT</span><strong>REAL NCMRWF TIGGE</strong><span>{String((tigge?.provenance["forecast"] as Record<string, unknown> | undefined)?.["ensemble_count"] ?? "Source")} members · source grid</span></div>
          <ArrowRight className="hidden self-center text-muted-foreground lg:block" aria-hidden="true" />
          <div className="metric-mini"><span>MODEL</span><strong>U-NET · EXPERIMENTAL</strong><span>Cached IMDAA snapshot; not matched to TIGGE</span></div>
          <ArrowRight className="hidden self-center text-muted-foreground lg:block" aria-hidden="true" />
          <div className="metric-mini"><span>MODEL</span><strong>DIFFUSION · EXPERIMENTAL</strong><span>Cached IMDAA snapshot; not matched to TIGGE</span></div>
          <ArrowRight className="hidden self-center text-muted-foreground lg:block" aria-hidden="true" />
          <div className="metric-mini"><span>OUTPUT</span><strong>MODEL-DERIVED HIGHER-RESOLUTION DEMONSTRATION</strong><span>2× spatial refinement only; no km target asserted</span></div>
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          <IntegrityItem label="Independent 5-km verification" value="NOT AVAILABLE" />
          <IntegrityItem label="Cached output timestamp" value={`IMDAA peak-anomaly snapshot · ${snapshotTime}`} />
          <IntegrityItem label="Output grid" value={`2× experimental spatial refinement · U-Net ${formatGridShape(unet)}; diffusion ${formatGridShape(diffusion)}`} />
          <IntegrityItem label="Matched TIGGE model output" value="NOT AVAILABLE" />
        </div>
      </div>
      <div className="mt-4 border-t border-border pt-3">
        <div className="data-label">Data provenance</div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          <IntegrityItem label="Forecast source" value="NCMRWF TIGGE · dems (New Delhi)" />
          <IntegrityItem label="Initialization" value={tigge?.forecast_initialization_time ?? "Loading TIGGE metadata…"} />
          <IntegrityItem label="Ensemble / lead steps" value={`${String((tigge?.provenance["forecast"] as Record<string, unknown> | undefined)?.["ensemble_count"] ?? "—")} members · ${tigge?.timeline.length ?? "—"} steps`} />
          <IntegrityItem label="Grid provenance" value={tigge ? `${String((tigge.provenance["forecast"] as Record<string, unknown> | undefined)?.["grid_dimensions"] ?? "Source grid unavailable")} source GRIB · ${tigge.grid_dimensions.latitude} × ${tigge.grid_dimensions.longitude} analysis domain` : "Loading TIGGE metadata…"} />
          <IntegrityItem label="Reference data" value="IMDAA · May 2011–2019" />
          <IntegrityItem label="Model output status" value="Cached experimental IMDAA snapshots; not matched to TIGGE timestep" />
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-border pt-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="data-label">TIGGE variable</span>
          <select className="rounded border border-border bg-card px-3 py-2" value={forecastVariable} onChange={(event) => setForecastVariable(event.target.value as typeof forecastVariable)}>
            <option value="t2m">Temperature · t2m (K)</option>
            <option value="rainfall">Accumulated precipitation · tp</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="data-label">TIGGE member</span>
          <select className="rounded border border-border bg-card px-3 py-2" disabled value="mean">
            <option value="mean">Ensemble mean · cached field</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <span className="data-label">Model focus</span>
          <select className="rounded border border-border bg-card px-3 py-2" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value as typeof selectedModel)}>
            <option value="unet">U-Net baseline</option>
            <option value="diffusion">Conditional diffusion</option>
          </select>
        </label>
        <Button variant="outline" onClick={() => {
          refreshCache.mutate({});
          void tiggeQuery.refetch();
          void tiggeFieldQuery.refetch();
        }} disabled={refreshCache.isPending}>
          {refreshCache.isPending ? <LoaderCircle className="animate-spin" /> : <RefreshCw />} Load/update cached fields
        </Button>
        <Button variant="ghost" onClick={() => { setSelectedModel("unet"); setTransectLatitude(suggestedTransect ?? null); setForecastIndex(0); setForecastVariable("t2m"); }}>Reset</Button>
      </div>
      <p className="mt-2 text-[10px] text-muted-foreground">
        TIGGE has {String((tigge?.provenance["forecast"] as Record<string, unknown> | undefined)?.["ensemble_count"] ?? "source")} source members; the selected input field is an ensemble mean. Member-specific arrays and lead/member-specific U-Net/diffusion outputs were not retained. Temperature is the only cached model comparison variable; rainfall downscaling is unavailable.
      </p>
    </section>
    {loading && <section className="panel p-5 text-sm text-muted-foreground">Loading cached downscaling metadata…</section>}
    {error && <section role="alert" className="panel mb-4 border border-destructive/40 p-4 text-sm">
      <div className="font-semibold text-destructive">{health.isSuccess ? "CACHED MODEL OUTPUT UNAVAILABLE" : "BACKEND OFFLINE"}</div>
      <p className="mt-2 text-muted-foreground">{error}</p>
      <Button className="mt-3" variant="outline" onClick={() => { void overviewQuery.refetch(); void health.refetch(); }}>Retry</Button>
    </section>}
    {replay && !error && <>
      <section className="panel mb-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="data-label">REAL NCMRWF TIGGE ~12 km INPUT · not connected to cached outputs</div>
            <div className="mt-1 font-mono text-xs">{forecastStep?.valid_time ?? "Loading TIGGE replay…"} · lead +{forecastStep?.lead_time_hours ?? "N/A"} h · {forecastVariable === "rainfall" ? "accumulated precipitation amount, not a rate" : "ensemble-mean 2 m temperature"}</div>
          </div>
          <span className="data-mode">{String((tigge?.provenance["forecast"] as Record<string, unknown> | undefined)?.["ensemble_count"] ?? "SOURCE")} MEMBERS · MEAN FIELD RETAINED</span>
        </div>
        {tigge && <input
          aria-label="TIGGE forecast lead"
          className="mt-4 w-full accent-primary"
          type="range"
          min={0}
          max={Math.max(0, tigge.timeline.length - 1)}
          value={forecastIndex}
          onChange={(event) => setForecastIndex(Number(event.target.value))}
        />}
        {tiggeQuery.isError
          ? <p role="alert" className="mt-3 text-sm text-amber-200">TIGGE context unavailable: {tiggeQuery.error.message}. The cached IMDAA model comparison remains independent.</p>
          : tiggeFieldQuery.isError
            ? <p role="alert" className="mt-3 text-sm text-amber-200">TIGGE field unavailable: {tiggeFieldQuery.error.message}</p>
            : tiggeFieldQuery.data
              ? <div className="mt-4">
                <GeographicField
                  field={tiggeFieldQuery.data}
                  title={forecastVariable === "rainfall" ? "TIGGE accumulated precipitation · ensemble mean" : "TIGGE 2 m temperature · ensemble mean"}
                  trackPoints={[]}
                  center={undefined}
                  bounds={undefined}
                  palette={forecastVariable === "t2m" ? "temperature" : "efi"}
                  geographicSource="NCMRWF TIGGE"
                />
                <p className="mt-2 text-[10px] text-muted-foreground">
                  Real TIGGE forecast field. This field is not fed into or directly compared with the static IMDAA U-Net/diffusion cache below.
                  {tiggeFieldQuery.data.visualization_downsampled ? ` Display sampling: ${tiggeFieldQuery.data.source_shape?.join("×") ?? "source"} to ${tiggeFieldQuery.data.returned_shape?.join("×") ?? "display"}.` : ""}
                </p>
              </div>
              : <p className="mt-3 text-sm text-muted-foreground">Loading cached TIGGE input field…</p>}
      </section>
      <section className="panel mb-4 p-4">
        <div className="data-label">Cached experimental outputs · not matched to the selected TIGGE timestep</div>
        <p className="mt-2 text-xs text-muted-foreground">These panels use the cached IMDAA historical crop and its experimental reconstructions, not the real TIGGE field shown above.</p>
      </section>
      <div className="grid gap-4 xl:grid-cols-3">
        {modelFields.map((item) => <div key={item.variable} className={item.variable === selectedModel ? "rounded border border-primary/70 ring-1 ring-primary/30" : ""}>
          <GeographicField field={item.field} title={item.variable === "crop_t2m" ? item.title : `${item.variable === "unet" ? "U-Net" : "Diffusion"} · Cached experimental model output — not matched to this TIGGE timestep`} trackPoints={[]} center={undefined} bounds={undefined} palette={item.palette} geographicSource="IMDAA" />
          {item.variable !== "crop_t2m" && <p className="mt-1 text-[10px] text-muted-foreground">2× experimental spatial refinement · source grid {formatGridShape(item.field)} · no kilometer resolution asserted.</p>}
        </div>)}
      </div>
      <section className="panel mt-4 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><div className="data-label">Model-derived Extreme Amplitude Diagnostic</div><p className="mt-1 text-xs text-muted-foreground">Descriptive diagnostics for the IMDAA input crop and cached outputs only. Not independent verification, accuracy, or forecast skill.</p></div>
          <span className="data-mode">{selectedModel === "unet" ? "U-NET BASELINE SELECTED" : "DIFFUSION SELECTED"}</span>
        </div>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[620px] text-left text-xs">
            <thead><tr className="border-b border-border text-muted-foreground"><th className="p-2">Diagnostic</th><th className="p-2">IMDAA input crop</th><th className="p-2">U-Net · cached</th><th className="p-2">Diffusion · cached</th></tr></thead>
            <tbody>
              <tr className="border-b border-border/60"><td className="p-2">Peak field value (K)</td>{peakValues.map((value, index) => <td key={index} className="p-2">{value?.toFixed(3) ?? "N/A"}</td>)}</tr>
              <tr className="border-b border-border/60"><td className="p-2">Upper-tail amplitude (p95 − p50, K)</td>{tailAmplitudes.map((value, index) => <td key={index} className="p-2">{value?.toFixed(3) ?? "N/A"}</td>)}</tr>
              <tr><td className="p-2">Upper-tail amplitude ratio vs input</td>{tailRatios.map((value, index) => <td key={index} className="p-2">{index === 0 ? "1.000" : value === undefined ? "N/A" : value.toFixed(3)}</td>)}</tr>
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel mt-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><div className="data-label">Geographic transect profile · TIGGE t2m / U-Net / Diffusion</div><p className="mt-1 text-xs text-muted-foreground">Nearest available latitude; profiles are sampled over shared geographic coverage. The Kelvin axis is zoomed to the plotted values to reveal spatial variation. Outputs shown are cached experimental snapshots and are not a matched forecast-time comparison.</p></div>
          <label className="flex items-center gap-2 text-xs"><span className="data-label">Latitude</span><input aria-label="Transect latitude" className="w-28 rounded border border-border bg-card px-2 py-1" type="number" step="0.001" value={currentTransectLatitude?.toFixed(3) ?? ""} onChange={(event) => setTransectLatitude(event.target.value === "" ? null : Number(event.target.value))} /></label>
        </div>
        <div className="mt-3 h-72">
          {profile.length
            ? <ResponsiveContainer width="100%" height="100%">
              <LineChart data={profile} margin={{ top: 8, right: 16, bottom: 12, left: 4 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="longitude" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(value: number) => `${value.toFixed(1)}°E`} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
                <YAxis domain={transectDomain} tickFormatter={(value: number) => value.toFixed(1)} tick={{ fill: "var(--muted-foreground)", fontSize: 10 }} />
                <ChartTooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", fontSize: 11 }} labelFormatter={(value) => `${Number(value).toFixed(2)}°E`} />
                <Legend />
                <Line type="linear" dataKey="tiggeInput" name="TIGGE input · t2m" stroke="#38bdf8" dot={false} isAnimationActive={false} connectNulls={false} />
                <Line type="linear" dataKey="unet" name="U-Net" stroke="#f59e0b" dot={false} isAnimationActive={false} connectNulls={false} />
                <Line type="linear" dataKey="diffusion" name="Diffusion" stroke="#a78bfa" dot={false} isAnimationActive={false} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
            : <div className="grid h-full place-items-center text-sm text-muted-foreground">Transect unavailable: TIGGE t2m and both cached model fields must share geographic coverage.</div>}
        </div>
        <p className="mt-2 text-[10px] text-muted-foreground">The TIGGE line is from the selected valid time; U-Net and diffusion lines are static IMDAA-derived cached snapshots. They are not a matched-input comparison, and neither model is assumed to perform better.</p>
      </section>
      <section className="panel mt-4 p-4">
        <div className="data-label">Model comparison · qualitative, not verification</div>
        <div className="mt-3 grid gap-3 md:grid-cols-[1fr_1fr_1.3fr]">
          <div className="metric-mini"><span>U-Net</span><strong>Deterministic regression baseline</strong><span>Cached experimental output</span></div>
          <div className="metric-mini"><span>Diffusion</span><strong>Conditional generative prototype</strong><span>Cached experimental output</span></div>
          <div className="metric-mini"><span>Scientific status</span><strong>EXPERIMENTAL · NOT VERIFIED</strong><span>Distributional sampling is not benchmarked against independent observations.</span></div>
        </div>
      </section>
      <section className="panel mt-4 border border-amber-500/40 p-4 text-sm">
        <div className="font-semibold text-amber-200">Scientific status: experimental demonstration.</div>
        <p className="mt-2 text-xs text-muted-foreground">
          These 2× experimental fields are generated from an IMDAA historical crop. They are not observed truth, are not TIGGE-matched forecast outputs, and have no independently verified kilometer-scale target. This is not operational downscaling.
        </p>
      </section>
    </>}
  </>;
}

function makeTransectProfile(
  tiggeInput: AmphanField | undefined,
  unet: AmphanField | undefined,
  diffusion: AmphanField | undefined,
  requestedLatitude: number | null,
) {
  if (!tiggeInput || !unet || !diffusion || tiggeInput.latitude.length < 2 || tiggeInput.longitude.length < 2) return [];
  const sharedSouth = Math.max(...[tiggeInput, unet, diffusion].map((field) => Math.min(...field.latitude)));
  const sharedNorth = Math.min(...[tiggeInput, unet, diffusion].map((field) => Math.max(...field.latitude)));
  const sharedWest = Math.max(...[tiggeInput, unet, diffusion].map((field) => Math.min(...field.longitude)));
  const sharedEast = Math.min(...[tiggeInput, unet, diffusion].map((field) => Math.max(...field.longitude)));
  if (sharedSouth > sharedNorth || sharedWest >= sharedEast) return [];
  const selectedLatitude = Math.min(sharedNorth, Math.max(sharedSouth, requestedLatitude ?? (sharedSouth + sharedNorth) / 2));
  const nearestRow = (field: AmphanField) => {
    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    field.latitude.forEach((latitude, index) => {
      const distance = Math.abs(latitude - selectedLatitude);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    return bestIndex;
  };
  const sampleLine = (field: AmphanField, targetLongitude: number) => {
    const row = field.values[nearestRow(field)];
    if (!row || row.length !== field.longitude.length) return null;
    const pairs = field.longitude
      .map((longitude, index) => ({ longitude, value: row[index] }))
      .filter((point): point is { longitude: number; value: number } => typeof point.value === "number" && Number.isFinite(point.value))
      .sort((a, b) => a.longitude - b.longitude);
    const first = pairs[0];
    const last = pairs[pairs.length - 1];
    if (!first || !last || targetLongitude < first.longitude || targetLongitude > last.longitude) return null;
    for (let index = 1; index < pairs.length; index += 1) {
      const right = pairs[index];
      const left = pairs[index - 1];
      if (!right || !left) continue;
      if (targetLongitude <= right.longitude) {
        const fraction = (targetLongitude - left.longitude) / (right.longitude - left.longitude);
        return left.value + fraction * (right.value - left.value);
      }
    }
    return last.value;
  };
  return tiggeInput.longitude.filter((longitude) => longitude >= sharedWest && longitude <= sharedEast).map((longitude) => ({
    longitude,
    tiggeInput: sampleLine(tiggeInput, longitude),
    unet: sampleLine(unet, longitude),
    diffusion: sampleLine(diffusion, longitude),
  }));
}

function formatGridShape(field: AmphanField | undefined) {
  const shape = field?.source_shape ?? field?.returned_shape;
  return shape?.length === 2 ? `${shape[0]} × ${shape[1]}` : "unavailable";
}

export function PhysicsPage() { const { currentResult } = useStormtrace(); return <><PageHeading eyebrow="Scientific validation" title="Physics Validation" description="Post-hoc ensemble agreement and moisture-convergence consistency, represented without claiming a hard physical guarantee." /><PhysicsGauge result={currentResult} /><div className="mt-4 grid gap-4 md:grid-cols-3">{[["Ensemble agreement","Generated field compared with the synthetic ensemble envelope."],["Conservation consistency","Moisture convergence consistency between coarse and generated fields."],["Model confidence","Severity confidence after physics-aware downgrade logic."]].map(([title,body]) => <section key={title} className="panel p-4"><CheckCircle2 className="mb-4 size-5 text-primary"/><h2 className="text-sm font-semibold">{title}</h2><p className="mt-2 text-xs leading-relaxed text-muted-foreground">{body}</p></section>)}</div></>; }

export function AlertCenterPage() {
  const [severity,setSeverity] = useState<Severity | "all">("all"); const [event,setEvent] = useState(""); const [search,setSearch] = useState(""); const [limit,setLimit] = useState(12); const [selected,setSelected] = useState<number|null>(null);
  const query = useAlerts({limit, ...(severity !== "all" ? {severity} : {}), ...(event ? {event_name:event} : {})});
  const rows = useMemo(() => (query.data ?? []).filter((a) => a.event_name.toLowerCase().includes(search.toLowerCase())),[query.data,search]);
  return <><PageHeading eyebrow="Persisted records" title="Alert Center" description="Backend-filtered alert records with authoritative detail retrieval." /><div className="mb-4 grid gap-2 md:grid-cols-[1fr_180px_220px_auto]"><div className="relative"><Search className="absolute left-3 top-2.5 size-4 text-muted-foreground"/><Input className="pl-9" value={search} onChange={(e)=>setSearch(e.target.value)} placeholder="Search loaded event names" /></div><Select value={severity} onValueChange={(v)=>setSeverity(v as Severity|"all")}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All severities</SelectItem><SelectItem value="low">Low</SelectItem><SelectItem value="moderate">Moderate</SelectItem><SelectItem value="severe">Severe</SelectItem></SelectContent></Select><Input value={event} onChange={(e)=>setEvent(e.target.value)} placeholder="Exact event filter"/><Button variant="outline" onClick={()=>query.refetch()}><RefreshCw/> Refresh</Button></div>{query.isError ? <ErrorState message="Alerts could not be loaded." retry={()=>query.refetch()} /> : rows.length ? <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{rows.map((alert)=><AlertCard key={alert.id} alert={alert} onOpen={()=>setSelected(alert.id)} />)}</div> : <section className="panel grid min-h-60 place-items-center text-center"><div><AlertCircle className="mx-auto mb-3 text-muted-foreground"/><h2 className="text-sm font-semibold">No alerts yet</h2><p className="mt-1 text-xs text-muted-foreground">Run an analysis or launch Historical Replay.</p></div></section>} {rows.length >= limit && limit < 200 && <Button className="mt-4" variant="outline" onClick={()=>setLimit((v)=>Math.min(200,v+12))}>Load more</Button>}<AlertDetail id={selected} close={()=>setSelected(null)} /></>;
}

export function VerificationPage() {
  const metrics = useMetrics(); const data = metrics.data ? Object.entries(metrics.data.by_severity).map(([name,value])=>({name,value})) : [];
  return <><PageHeading eyebrow="Backend aggregates" title="Model Verification" description="Only currently exposed evidence is shown. Tracking recall and false-alarm rate remain unavailable from the API." action={<Button variant="outline" onClick={()=>metrics.refetch()}><RefreshCw/> Refresh</Button>} />{metrics.isError ? <ErrorState message="Verification metrics could not be loaded." retry={()=>metrics.refetch()} /> : <div className="grid gap-4 lg:grid-cols-[.7fr_1.3fr]"><div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1"><MetricCard label="Total alerts" value={metrics.data?.total_alerts}/><MetricCard label="Average physics score" value={metrics.data?.avg_physics_score}/><MetricCard label="Avg diffusion vs U-Net gain" value={metrics.data ? `${metrics.data.avg_diffusion_vs_unet_amplitude_gain_pct.toFixed(2)}%` : undefined}/></div><section className="panel p-5"><div className="data-label">Severity distribution</div><div className="mt-4 h-72">{data.length ? <ResponsiveContainer width="100%" height="100%"><BarChart data={data}><CartesianGrid stroke="var(--border)" vertical={false}/><XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={10}/><YAxis allowDecimals={false} stroke="var(--muted-foreground)" fontSize={10}/><ChartTooltip contentStyle={{background:"var(--card)",border:"1px solid var(--border)"}}/><Bar dataKey="value" fill="var(--primary)" radius={[2,2,0,0]}/></BarChart></ResponsiveContainer> : <EmptyResult />}</div></section></div>}</>;
}

export function SystemPage() {
  const health = useBackendHealth(); const storm = useStormtrace(); const sync = useSyncPipeline();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const connected = mounted && health.isSuccess;
  const sourceLabel = getSourceLabel(health.data?.source_label ?? health.data?.data_source);
  const endpoints = ["GET /health","POST /pipeline/run","GET /pipeline/status/{job_id}","POST /pipeline/run_sync","POST /replay/{event_key}","GET /replay/events","GET /alerts/latest","GET /alerts/{alert_id}","GET /metrics"];
  return <><PageHeading eyebrow="Developer operations" title="System / API" description="Connection diagnostics, in-session jobs, endpoint catalog and raw pipeline payloads." /><div className="grid gap-4 xl:grid-cols-[.8fr_1.2fr]"><section className="panel p-5"><div className="data-label">Connection</div><div className="mt-4 space-y-3"><SystemRow label="Backend URL" value={API_BASE_URL}/><SystemRow label="Status" value={connected ? "Connected" : "Unreachable"}/><SystemRow label="API version" value={mounted ? health.data?.version ?? "—" : "—"}/><SystemRow label="Data source" value={mounted ? sourceLabel : "—"}/><SystemRow label="Health message" value={mounted ? health.data?.status ?? "—" : "—"}/></div><Button className="mt-5" variant="outline" onClick={()=>health.refetch()}><RefreshCw/> Retry connection</Button></section><section className="panel p-5"><div className="data-label">Recent jobs · current browser session</div><div className="mt-4 space-y-2">{storm.jobs.length ? storm.jobs.map((job)=><div key={job.id} className="grid grid-cols-[1fr_auto] gap-3 border-b border-border py-2 text-xs"><div><div className="font-mono">{job.id}</div><div className="mt-1 text-[10px] uppercase text-muted-foreground">{displayEvent(job.eventName)} · {job.mode}</div></div><span className="font-mono uppercase text-primary">{job.status}</span></div>) : <EmptyResult />}</div></section></div><div className="mt-4 grid gap-4 xl:grid-cols-[.8fr_1.2fr]"><section className="panel p-5"><div className="data-label">Endpoints available</div><div className="mt-4 space-y-2 font-mono text-[10px]">{endpoints.map((e)=><div key={e} className="border-l-2 border-primary/30 bg-muted/30 p-2">{e}</div>)}</div><div className="mt-6 border border-destructive/30 bg-destructive/5 p-4"><div className="flex gap-2 text-xs font-bold text-destructive"><ShieldAlert className="size-4"/> Developer / Advanced</div><p className="my-3 text-xs text-muted-foreground">Blocking execution — not recommended for normal dashboard operation.</p><Button variant="destructive" size="sm" disabled={sync.isPending} onClick={()=>sync.mutate({event_name:"live_run",seed:null},{onSuccess:(result)=>toast.success(`Sync alert #${result.alert_id} complete`)})}>{sync.isPending ? <LoaderCircle className="animate-spin"/> : <Play/>} Run synchronous pipeline</Button></div></section><JsonPanel value={sync.data ?? storm.currentResult} /></div></>;
}

function JsonPanel({value}:{value:unknown}) { const text=value?JSON.stringify(value,null,2):"No completed pipeline payload in this session."; return <section className="panel overflow-hidden"><div className="flex items-center justify-between border-b border-border p-3"><div className="flex items-center gap-2 text-xs font-semibold"><Braces className="size-4 text-primary"/> Actual JSON response</div><Button variant="ghost" size="sm" onClick={()=>{navigator.clipboard.writeText(text);toast.success("JSON COPIED")}} disabled={!value}><Clipboard/> Copy JSON</Button></div><pre className="max-h-[520px] overflow-auto p-4 font-mono text-[10px] leading-relaxed text-muted-foreground">{text}</pre></section> }
function AlertCard({alert,onOpen}:{alert:Alert;onOpen:()=>void}) { return <button onClick={onOpen} className="panel p-4 text-left transition-colors hover:border-primary/50"><div className="flex justify-between"><span className={`severity severity-${alert.severity}`}>{severityLabel(alert.severity)}</span><span className="font-mono text-[10px] text-muted-foreground">ALERT #{alert.id}</span></div><h2 className="mt-4 font-display text-lg font-semibold">{displayEvent(alert.event_name)}</h2><div className="mt-4 grid grid-cols-2 gap-3 text-xs">{[["Peak",alert.peak_value],["EFI",alert.efi_peak],["Confidence",alert.confidence],["Physics",alert.physics_score]].map(([k,v])=><div key={k as string}><span className="data-label">{k}</span><div className="mt-1 font-mono">{v}</div></div>)}</div><div className="mt-4 flex justify-between border-t border-border pt-3 text-[10px] text-muted-foreground"><span>Mesh [{alert.centroid_row}, {alert.centroid_col}]</span><span>{alert.radius_km} km radius</span></div></button> }
function AlertDetail({id,close}:{id:number|null;close:()=>void}) { const detail=useAlert(id); return <Dialog open={id!==null} onOpenChange={(o)=>!o&&close()}><DialogContent className="max-w-2xl"><DialogHeader><DialogTitle>Authoritative alert record</DialogTitle><DialogDescription>Fetched directly from GET /alerts/{id}</DialogDescription></DialogHeader>{detail.isLoading?<LoaderCircle className="mx-auto animate-spin"/>:detail.data?<><AlertCard alert={detail.data} onOpen={()=>{}}/><JsonPanel value={detail.data}/></>:<ErrorState message="Alert record unavailable." retry={()=>detail.refetch()}/>}</DialogContent></Dialog> }
function ErrorState({message,retry}:{message:string;retry:()=>unknown}) { return <section className="panel grid min-h-48 place-items-center p-6 text-center"><div><CloudOff className="mx-auto mb-3 text-destructive"/><div className="text-sm font-semibold">STORMTRACE ENGINE UNREACHABLE</div><p className="my-2 text-xs text-muted-foreground">{message}</p><Button size="sm" variant="outline" onClick={()=>retry()}><RefreshCw/> Retry connection</Button></div></section> }
function EmptyResult(){return <div className="grid h-full min-h-24 place-items-center text-xs text-muted-foreground"><span>Awaiting a completed pipeline result.</span></div>}
function MetricCard({label,value}:{label:string;value:string|number|undefined}){return <section className="metric-cell border border-border"><div className="data-label">{label}</div><div className="mt-3 font-mono text-3xl">{value ?? "—"}</div></section>}
function SystemRow({label,value}:{label:string;value:string}){return <div className="grid grid-cols-[110px_1fr] gap-3 text-xs"><span className="text-muted-foreground">{label}</span><strong className="truncate font-mono">{value}</strong></div>}
function FieldPattern({strength,diffusion}:{strength?:number|undefined;diffusion?:boolean}){return <div className={`field-pattern ${diffusion?"field-diffusion":""}`}><div className="field-core" style={{transform:`scale(${strength?Math.min(1.4,.8+Math.abs(strength)%1):.8})`}}/><span>DEMO VISUALIZATION</span></div>}
function HonestyNote({ sourceLabel }: { sourceLabel: string }) { return <div className="mb-4 flex gap-3 border border-status-warn/30 bg-status-warn/5 p-4 text-xs text-muted-foreground"><Database className="size-4 shrink-0 text-status-warn"/><span><strong className="text-foreground">DATA SOURCE: {sourceLabel.toUpperCase()}.</strong> This provider is configured in the backend and does not require NEPS-G credentials.</span></div> }
function metricDescription(label:string){return ({"EFI peak":"Maximum Extreme Forecast Index returned by the pipeline.","Diffusion peak":"Maximum conditional diffusion output.","U-Net peak":"Maximum regression baseline output.","Amplitude gain":"Diffusion peak relative to U-Net peak.","Physics score":"Combined ensemble agreement and moisture-convergence consistency.","Tracked points":"Mesh points contained in the 4D bounding box.","Forecast window":"Backend bounding-box timestep range mapped to D+3 through D+10.","Processing time":"Measured backend wall-clock runtime."} as Record<string,string>)[label] ?? label}