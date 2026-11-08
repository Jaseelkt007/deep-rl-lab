"use client";

import dynamic from "next/dynamic";

const Compare = dynamic(() => import("@/components/Compare"), { ssr: false });

export default function ComparePage() {
  return <Compare />;
}
