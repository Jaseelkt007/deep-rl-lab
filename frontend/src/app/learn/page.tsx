"use client";

import dynamic from "next/dynamic";
import NeedsBackend from "@/components/NeedsBackend";
import { STATIC } from "@/lib/api";

const LearnPage = dynamic(() => import("@/components/learn/LearnPage"), { ssr: false });

export default function Learn() {
  return STATIC ? <NeedsBackend active="learn" /> : <LearnPage />;
}
