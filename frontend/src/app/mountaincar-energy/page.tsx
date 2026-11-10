"use client";

import dynamic from "next/dynamic";

// MountainCar with the energy bonus in training: the same Run page, reading runs/mountaincar_energy/.
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function MountainCarEnergy() {
  return <RunPage envKey="mountaincar_energy" />;
}
