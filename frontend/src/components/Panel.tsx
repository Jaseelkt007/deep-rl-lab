"use client";

import { ReactNode, useState } from "react";

/** A white card with a small title and a "?" that explains what you are looking at. */
export default function Panel({
  title,
  help,
  children,
  className = "",
  right,
}: {
  title: string;
  help: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}) {
  return (
    <section className={`rounded-lg border border-neutral-200 bg-white p-3 ${className}`}>
      <header className="mb-2 flex items-center gap-1.5">
        <h2 className="text-[13px] font-medium text-neutral-800">{title}</h2>
        <Help text={help} />
        <div className="ml-auto flex items-center gap-2">{right}</div>
      </header>
      {children}
    </section>
  );
}

export function Help({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-label="What am I looking at?"
        onClick={() => setOpen((o) => !o)}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-neutral-300 text-[10px] leading-none text-neutral-500 hover:border-neutral-500 hover:text-neutral-800"
      >
        ?
      </button>
      {open && (
        <span className="absolute left-5 top-0 z-30 w-72 rounded-md border border-neutral-200 bg-white p-2.5 text-[12px] leading-snug text-neutral-700 shadow-lg">
          {text}
        </span>
      )}
    </span>
  );
}

/** Small chart title used inside panels. */
export function ChartTitle({ children, help }: { children: ReactNode; help?: string }) {
  return (
    <div className="mb-0.5 flex items-center gap-1 text-[11px] font-medium text-neutral-600">
      {children}
      {help && <Help text={help} />}
    </div>
  );
}
