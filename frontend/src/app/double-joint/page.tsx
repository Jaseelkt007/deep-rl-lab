"use client";

import dynamic from "next/dynamic";

// The double cart-pole with the joint rule: the same Run page, reading runs/double_cartpole_joint/.
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function DoubleCartPoleJoint() {
  return <RunPage envKey="double_cartpole_joint" />;
}
