"use client";

import dynamic from "next/dynamic";

const TerminalRun = dynamic(() => import("@/components/TerminalRun"), { ssr: false });

export default function TerminalPage() {
  return <TerminalRun />;
}
