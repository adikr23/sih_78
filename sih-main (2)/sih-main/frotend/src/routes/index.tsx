import { createFileRoute } from "@tanstack/react-router";
import { CommandCenterPage } from "@/components/pages/dashboard-pages";

// No head() here: the home route inherits title/description/og/twitter from
// __root.tsx, and ships no og:image so serve-time hosting can inject the
// project's social preview (explicit og:image or latest screenshot).
export const Route = createFileRoute("/")({
  head: () => ({ meta: [
    { title: "Command Center — STORMTRACE-GNN" },
    { name: "description", content: "Extreme-weather anomaly command center with scientific pipeline monitoring." },
    { property: "og:title", content: "Command Center — STORMTRACE-GNN" },
    { property: "og:description", content: "Extreme-weather anomaly command center with scientific pipeline monitoring." },
    { property: "og:type", content: "website" },
    { name: "twitter:card", content: "summary_large_image" },
  ] }),
  component: CommandCenterPage,
});
