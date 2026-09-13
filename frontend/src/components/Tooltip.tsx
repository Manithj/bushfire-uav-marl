import type { ReactNode } from "react";

export function Tip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="group relative inline-flex items-center gap-1">
      {children}
      <span className="pointer-events-none absolute bottom-[125%] left-0 z-30 hidden w-72 rounded border border-ink-600 bg-ink-900 p-2 text-[11px] leading-snug text-paper shadow-lg group-hover:block">
        {label}
      </span>
    </span>
  );
}
