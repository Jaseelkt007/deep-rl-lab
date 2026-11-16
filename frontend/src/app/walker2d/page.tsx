"use client";

import dynamic from "next/dynamic";

// Walker2d (MuJoCo, continuous actions): runs/walker2d/ written by scripts/train_continuous.py.
const WalkerPage = dynamic(() => import("@/components/WalkerPage"), { ssr: false });

export default function Walker2d() {
  return <WalkerPage />;
}
