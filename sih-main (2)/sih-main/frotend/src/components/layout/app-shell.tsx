import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";
import {
  Activity, AlertTriangle, Atom, Braces, ChevronLeft, CloudLightning, FlaskConical,
  Gauge, History, Menu, Moon, Network, Play, Radar, Sun, Waves,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useBackendHealth } from "@/hooks/use-stormtrace";
import { cn } from "@/lib/utils";
import { useStormtrace } from "@/components/stormtrace-provider";

const nav = [
  ["Command Center", "/", Radar], ["Live Analysis", "/live-analysis", Activity],
  ["IMDAA Replay", "/historical-replay", History], ["TIGGE Forecast", "/tigge-replay", CloudLightning], ["Anomaly Tracking", "/anomaly-tracking", Network],
  ["Downscaling Lab", "/downscaling-lab", Waves], ["Physics Validation", "/physics-validation", Atom],
  ["Alert Center", "/alert-center", AlertTriangle], ["Verification", "/verification", Gauge],
  ["API / System", "/system", Braces],
] as const;

function ThemeToggle() {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    const stored = window.localStorage.getItem("stormtrace-theme");
    const next = stored !== "light";
    setDark(next); document.documentElement.classList.toggle("dark", next);
  }, []);
  return <Button variant="ghost" size="icon" aria-label="Toggle color theme" onClick={() => { const next = !dark; setDark(next); document.documentElement.classList.toggle("dark", next); window.localStorage.setItem("stormtrace-theme", next ? "dark" : "light"); }}>{dark ? <Sun /> : <Moon />}</Button>;
}

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const path = useRouterState({ select: (state) => state.location.pathname });
  const health = useBackendHealth();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const { mode, setMode, currentJob, startReplay } = useStormtrace();
  const online = mounted && health.isSuccess;
  return (
    <TooltipProvider delayDuration={300}>
      <div className="min-h-screen bg-background text-foreground">
        <div className="atmosphere" aria-hidden />
        <aside className={cn("fixed inset-y-0 left-0 z-40 flex border-r border-border bg-sidebar transition-[width,transform] duration-200", collapsed ? "w-18" : "w-60", mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0")}>
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex h-16 items-center gap-3 border-b border-border px-4">
              <div className="grid size-9 shrink-0 place-items-center border border-primary/40 bg-primary/10 text-primary"><CloudLightning className="size-5" /></div>
              {!collapsed && <div className="min-w-0"><div className="font-display text-sm font-bold tracking-normal">STORMTRACE</div><div className="text-[9px] uppercase text-muted-foreground">GNN Command</div></div>}
            </div>
            <nav className="flex-1 space-y-1 overflow-y-auto p-2" aria-label="Primary navigation">
              {nav.map(([label, to, Icon]) => {
                const active = path === to;
                return <Tooltip key={to}><TooltipTrigger asChild><Link to={to} onClick={() => setMobileOpen(false)} className={cn("flex h-10 items-center gap-3 border-l-2 px-3 text-[11px] font-semibold uppercase transition-colors", active ? "border-primary bg-primary/10 text-primary" : "border-transparent text-muted-foreground hover:bg-accent hover:text-foreground")}><Icon className="size-4 shrink-0" />{!collapsed && <span>{label}</span>}</Link></TooltipTrigger>{collapsed && <TooltipContent side="right">{label}</TooltipContent>}</Tooltip>;
              })}
            </nav>
            <div className="space-y-3 border-t border-border p-3 text-[10px]">
              {!collapsed && <><StatusLine label="Backend" value={online ? "Online" : "Offline"} active={online} /><StatusLine label="Model" value={currentJob?.status ?? "Standby"} active={currentJob?.status === "running"} /><StatusLine label="Data mode" value={health.data?.data_mode ?? "Unknown"} active={health.data?.forecast_data === "real"} /></>}
              <Button variant="ghost" size="icon" className="hidden lg:inline-flex" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => setCollapsed((value) => !value)}><ChevronLeft className={cn("transition-transform", collapsed && "rotate-180")} /></Button>
            </div>
          </div>
        </aside>

        <div className={cn("transition-[padding] duration-200", collapsed ? "lg:pl-18" : "lg:pl-60")}>
          <header className="mesh-header sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/90 px-3 backdrop-blur-md sm:px-5">
            <div className="flex items-center gap-3"><Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen((v) => !v)} aria-label="Open navigation"><Menu /></Button><div><div className="font-display text-sm font-bold">STORMTRACE-GNN</div><div className="hidden text-[9px] uppercase text-muted-foreground sm:block">Extreme weather intelligence</div></div></div>
            <div className="hidden items-center gap-1 border border-border bg-muted/40 p-1 md:flex" aria-label="Operating mode"><Button type="button" variant="ghost" size="sm" onClick={() => setMode("live")} className={cn("h-6 rounded-none px-3 text-[10px] font-bold uppercase", mode === "live" ? "bg-primary text-primary-foreground hover:bg-primary/90" : "text-muted-foreground")}>Live</Button><Button type="button" variant="ghost" size="sm" onClick={() => setMode("replay")} className={cn("h-6 rounded-none px-3 text-[10px] font-bold uppercase", mode === "replay" ? "bg-primary text-primary-foreground hover:bg-primary/90" : "text-muted-foreground")}>Historical replay</Button></div>
            <div className="flex items-center gap-2"><div className={cn("hidden items-center gap-2 text-[10px] font-bold uppercase sm:flex", online ? "text-status-ok" : "text-destructive")}><span className={cn("size-1.5 rounded-full", online ? "bg-status-ok" : "bg-destructive")} />{online ? "System online" : "Backend offline"}</div><Button size="sm" onClick={() => startReplay()}><Play /> <span className="hidden sm:inline">Launch demo</span></Button><ThemeToggle /></div>
          </header>
          <main className="relative mx-auto min-h-[calc(100vh-4rem)] max-w-[1920px] p-3 sm:p-5">{children}</main>
        </div>
      </div>
    </TooltipProvider>
  );
}

function StatusLine({ label, value, active }: { label: string; value: string; active?: boolean }) {
  return <div className="flex items-center justify-between gap-2"><span className="uppercase text-muted-foreground">{label}</span><span className={cn("truncate font-mono uppercase", active && "text-status-ok")}>{value}</span></div>;
}

export function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="mb-5 flex flex-col justify-between gap-3 border-b border-border pb-4 md:flex-row md:items-end"><div><div className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase text-primary"><FlaskConical className="size-3" />{eyebrow}</div><h1 className="font-display text-2xl font-semibold sm:text-3xl">{title}</h1><p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p></div>{action}</div>;
}