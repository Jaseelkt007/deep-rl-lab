"use client";

import dynamic from "next/dynamic";

// MountainCar (official MountainCar-v0): the same Run page, reading runs/mountaincar/.
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function MountainCar() {
  return <RunPage envKey="mountaincar" />;
}
