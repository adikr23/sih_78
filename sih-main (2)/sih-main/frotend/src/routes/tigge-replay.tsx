import { createFileRoute } from "@tanstack/react-router";
import { TiggeReplayPage } from "@/components/pages/dashboard-pages";

export const Route = createFileRoute("/tigge-replay")({
  head: () => ({
    meta: [
      { title: "NCMRWF TIGGE Amphan Replay — STORMTRACE-GNN" },
      { name: "description", content: "Cached NCMRWF TIGGE Amphan forecast replay with experimental geographic anomaly tracking." },
    ],
  }),
  component: TiggeReplayPage,
});
