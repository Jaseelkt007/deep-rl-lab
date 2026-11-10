"use client";

import dynamic from "next/dynamic";

// LunarLander: the same Run page, reading runs/lunarlander/ (or the saved copy on the hosted site).
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function LunarLander() {
  return <RunPage envKey="lunarlander" />;
}
