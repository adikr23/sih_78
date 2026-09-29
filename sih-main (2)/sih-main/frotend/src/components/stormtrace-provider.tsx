import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import { usePipelineJob, useStartPipeline } from "@/hooks/use-stormtrace";
import type { JobRecord, OperatingMode, PipelineJobStatus, PipelineResult } from "@/lib/types";

interface StormtraceContextValue {
  mode: OperatingMode;
  setMode: (mode: OperatingMode) => void;
  currentJob: PipelineJobStatus | null;
  currentResult: PipelineResult | undefined;
  selectedReplay: string;
  setSelectedReplay: (event: string) => void;
  jobs: JobRecord[];
  startLive: () => void;
  startReplay: (event?: string) => void;
  isStarting: boolean;
  startError: Error | null;
}

const StormtraceContext = createContext<StormtraceContextValue | null>(null);

export function StormtraceProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<OperatingMode>("replay");
  const [currentJob, setCurrentJob] = useState<PipelineJobStatus | null>(null);
  const [currentResult, setCurrentResult] = useState<PipelineResult>();
  const [selectedReplay, setSelectedReplay] = useState("cyclone_amphan");
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const handledTerminalJobs = useRef(new Set<string>());

  const terminal = useCallback((job: PipelineJobStatus) => {
    setCurrentJob(job);
    if (job.status === "done" && job.result) {
      setCurrentResult(job.result);
      toast.success("PIPELINE COMPLETE", { description: `Alert #${job.result.alert_id} persisted` });
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void queryClient.invalidateQueries({ queryKey: ["metrics"] });
    } else if (job.status === "failed") toast.error("PIPELINE FAILED");
    setJobs((items) => items.map((item) => item.id === job.job_id ? { ...item, status: job.status, completedAt: new Date().toISOString(), ...(job.result ? { result: job.result } : {}) } : item));
  }, [queryClient]);

  const polled = usePipelineJob(currentJob);
  useEffect(() => {
    const job = polled.data;
    if (!job) return;
    setCurrentJob(job);
    if ((job.status === "done" || job.status === "failed") && !handledTerminalJobs.current.has(job.job_id)) {
      handledTerminalJobs.current.add(job.job_id);
      terminal(job);
    }
  }, [polled.data, terminal]);

  const started = useStartPipeline((job, nextMode, eventName) => {
    setMode(nextMode); setCurrentJob(job);
    setJobs((items) => [{ id: job.job_id, mode: nextMode, eventName, startedAt: new Date().toISOString(), status: job.status }, ...items].slice(0, 8));
    toast.success(nextMode === "replay" ? "REPLAY STARTED" : "ANALYSIS STARTED", { description: `Job ${job.job_id.slice(0, 8)}` });
  });

  const value = useMemo(() => ({
    mode, setMode, currentJob, currentResult, selectedReplay, setSelectedReplay, jobs,
    startLive: () => started.mutate({ input: { event_name: "live_run", seed: null }, mode: "live" }),
    startReplay: (event = selectedReplay) => started.mutate({ input: { event_name: event, seed: null }, mode: "replay", eventKey: event }),
    isStarting: started.isPending,
    startError: started.error,
  }), [mode, currentJob, currentResult, selectedReplay, jobs, started]);

  return <StormtraceContext.Provider value={value}>{children}</StormtraceContext.Provider>;
}

export function useStormtrace() {
  const context = useContext(StormtraceContext);
  if (!context) throw new Error("useStormtrace must be used inside StormtraceProvider");
  return context;
}