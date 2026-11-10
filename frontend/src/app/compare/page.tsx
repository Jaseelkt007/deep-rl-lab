"use client";

import dynamic from "next/dynamic";
import NeedsBackend from "@/components/NeedsBackend";
import { STATIC } from "@/lib/api";

const Compare = dynamic(() => import("@/components/Compare"), { ssr: false });

export default function ComparePage() {
  return STATIC ? <NeedsBackend active="compare" /> : <Compare />;
}
