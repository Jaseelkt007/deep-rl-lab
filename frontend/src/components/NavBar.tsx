"use client";

import Link from "next/link";
import { ReactNode } from "react";
import { STATIC } from "@/lib/api";

const PAGES = [
  { href: "/", key: "run", label: "Run" },
  { href: "/learn", key: "learn", label: "Learn", live: true },
  { href: "/compare", key: "compare", label: "Compare", live: true },
] as const;

/** The top bar shared by every page. `right` holds page-specific controls. */
export default function NavBar({ active, right }: { active: (typeof PAGES)[number]["key"]; right?: ReactNode }) {
  return (
    <header className="sticky top-0 z-20 flex items-center gap-4 border-b border-neutral-200 bg-white/90 px-4 py-2 backdrop-blur">
      <span className="text-[14px] font-semibold text-neutral-900">Deep RL Lab</span>
      <nav className="flex gap-3 text-[12px]">
        {PAGES.filter((p) => !(STATIC && "live" in p)).map((p) => (
          <Link key={p.key} href={p.href} className={p.key === active ? "font-medium text-neutral-900" : "text-neutral-500 hover:text-neutral-900"}>
            {p.label}
          </Link>
        ))}
      </nav>
      {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
    </header>
  );
}
