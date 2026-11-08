"use client";

import dynamic from "next/dynamic";

// The dashboard draws on canvas and opens a WebSocket: render it in the browser only.
const Dashboard = dynamic(() => import("@/components/Dashboard"), { ssr: false });

export default function Home() {
  return <Dashboard />;
}
