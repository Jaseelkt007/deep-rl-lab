"use client";

import dynamic from "next/dynamic";

// LunarLander with wind and gusts: the same Run page, reading runs/lunarlander_wind/.
const RunPage = dynamic(() => import("@/components/RunPage"), { ssr: false });

export default function LunarLanderWind() {
  return <RunPage envKey="lunarlander_wind" />;
}
