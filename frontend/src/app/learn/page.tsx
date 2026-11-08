"use client";

import dynamic from "next/dynamic";

const LearnPage = dynamic(() => import("@/components/learn/LearnPage"), { ssr: false });

export default function Learn() {
  return <LearnPage />;
}
