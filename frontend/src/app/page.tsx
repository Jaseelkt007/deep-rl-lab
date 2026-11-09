"use client";

import dynamic from "next/dynamic";

// The Run page: the latest training run. Rendered in the browser only (canvas, charts).
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function Home() {
  return <RunPage />;
}
