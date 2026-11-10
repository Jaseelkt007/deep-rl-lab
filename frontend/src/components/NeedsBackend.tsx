"use client";

import NavBar from "@/components/NavBar";

/** Shown on the hosted site for pages that train or step a network live (they need the Python backend). */
export default function NeedsBackend({ active }: { active: "learn" | "compare" }) {
  return (
    <div className="min-h-screen bg-[#fafaf9]">
      <NavBar active={active} />
      <main className="mx-auto max-w-[720px] p-6 text-[13px] leading-relaxed text-neutral-700">
        <p className="font-medium text-neutral-900">This page runs the training live, so it needs the local backend.</p>
        <p className="mt-2">
          The hosted site shows saved runs only. To use this page, clone the repository and start the backend and the
          web app as the README describes.
        </p>
      </main>
    </div>
  );
}
