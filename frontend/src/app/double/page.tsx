"use client";

import dynamic from "next/dynamic";

// The double cart-pole: the same Run page, reading runs/double_cartpole/. Browser only (canvas, charts).
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function DoubleCartPole() {
  return <RunPage envKey="double_cartpole" />;
}
