import { queryOptions, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AlertsQuery, OperatingMode, PipelineJobStatus, PipelineRunRequest } from "@/lib/types";
import { toast } from "sonner";

export const healthQueryOptions = queryOptions({ queryKey: ["health"], queryFn: ({ signal }) => api.health(signal), retry: 2, refetchInterval: 15_000 });
export const replayEventsQueryOptions = queryOptions({ queryKey: ["replay-events"], queryFn: ({ signal }) => api.replayEvents(signal), retry: 2, staleTime: 60_000 });
export const metricsQueryOptions = queryOptions({ queryKey: ["metrics"], queryFn: ({ signal }) => api.metrics(signal), retry: 2 });

export function useBackendHealth() { return useQuery(healthQueryOptions); }
export function useReplayEvents() { return useQuery(replayEventsQueryOptions); }
export function useMetrics() { return useQuery(metricsQueryOptions); }
export function useAlerts(query: AlertsQuery) {
  return useQuery({ queryKey: ["alerts", query], queryFn: ({ signal }) => api.alerts(query, signal), retry: 2 });
}
export function useAlert(id: number | null) {
  return useQuery({ queryKey: ["alert", id], queryFn: ({ signal }) => api.alert(id ?? 0, signal), enabled: id !== null, retry: 1 });
}

export function usePipelineJob(job: PipelineJobStatus | null) {
  return useQuery({
    queryKey: ["pipeline-job", job?.job_id],
    queryFn: ({ signal }) => api.getJob(job?.job_id ?? "", signal),
    enabled: Boolean(job?.job_id && job.status !== "done" && job.status !== "failed"),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === "done" || data?.status === "failed") return false;
      return 1_500;
    },
    retry: (count, error) => error instanceof Error && error.message.includes("not found") ? count < 4 : count < 2,
  });
}

export function useStartPipeline(onStarted: (job: PipelineJobStatus, mode: OperatingMode, eventName: string) => void) {
  return useMutation({ mutationFn: ({ input, mode, eventKey }: { input: PipelineRunRequest; mode: OperatingMode; eventKey?: string }) => mode === "replay" ? api.runReplay(eventKey ?? input.event_name) : api.runPipeline(input), onSuccess: (job, vars) => onStarted(job, vars.mode, vars.eventKey ?? vars.input.event_name), onError: () => toast.error("REQUEST FAILED", { description: "The STORMTRACE engine could not start this run." }) });
}

export function useSyncPipeline() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (input: PipelineRunRequest) => api.runSync(input), onSuccess: () => { void client.invalidateQueries({ queryKey: ["alerts"] }); void client.invalidateQueries({ queryKey: ["metrics"] }); } });
}