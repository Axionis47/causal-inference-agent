/** The atoms. Every screen is built from these, so the terminal look is
 *  enforced by construction rather than by remembering to type the classes.
 *
 *  Carried over from the original app's aesthetics spec:
 *    - the 6px status dot, always paired with a label, amber only when live
 *    - the [ BRACKETED ] uppercase mono label, for captions only, never links
 *    - borders define surfaces; no shadows anywhere
 */
import type { ReactNode } from "react";

export type Status = "pending" | "live" | "ok" | "failed" | "info";

const DOT: Record<Status, string> = {
  pending: "bg-edge-subtle",
  live: "bg-amber animate-pulse-live", // amber-only: it is the heartbeat
  ok: "bg-mint",
  failed: "bg-rose",
  info: "bg-indigo",
};

export function Dot({ status }: { status: Status }) {
  return <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${DOT[status]}`} />;
}

/** Pane caption. Brackets are part of the string, not a border. */
export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-2xs font-mono text-ink-tertiary uppercase tracking-label">
      [ {children} ]
    </span>
  );
}

export function Pane({
  caption,
  right,
  children,
  className = "",
}: {
  caption: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`border border-edge-subtle bg-canvas-raised ${className}`}>
      <header className="flex items-center justify-between px-3 py-2 border-b border-edge-subtle">
        <Label>{caption}</Label>
        {right}
      </header>
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  tone = "default",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  tone?: "default" | "primary";
}) {
  const base =
    "px-3 py-1.5 text-2xs font-mono uppercase tracking-label border transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const tones = {
    default: "border-edge text-ink-secondary hover:text-ink hover:border-edge-strong",
    primary: "border-amber-dim text-amber hover:bg-amber hover:text-ink-inverse",
  };
  return (
    <button className={`${base} ${tones[tone]}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export function Field({
  label,
  value,
  onChange,
  placeholder,
  rows = 1,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
}) {
  const shared =
    "w-full bg-canvas-inset border border-edge-subtle px-3 py-2 text-sm text-ink placeholder:text-ink-tertiary focus:outline-none focus:border-edge-strong";
  return (
    <label className="block space-y-1.5">
      <Label>{label}</Label>
      {rows > 1 ? (
        <textarea
          className={`${shared} resize-none`}
          rows={rows}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <input
          className={shared}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </label>
  );
}

/** A key/value row. Values are mono and tabular so columns line up. */
export function Row({ k, v, tone }: { k: string; v: ReactNode; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 border-b border-edge-subtle/60 last:border-0">
      <span className="text-xs text-ink-secondary shrink-0">{k}</span>
      <span className={`text-xs font-mono tabular text-right ${tone ?? "text-ink"}`}>{v}</span>
    </div>
  );
}

/** Step rail across the top. The current step is live, past steps settled. */
export function Steps({ at }: { at: number }) {
  const names = ["ask", "data", "design", "running", "result"];
  return (
    <nav className="flex items-center gap-4">
      {names.map((n, i) => (
        <span key={n} className="flex items-center gap-1.5">
          <Dot status={i < at ? "ok" : i === at ? "live" : "pending"} />
          <span
            className={`text-2xs font-mono uppercase tracking-label ${
              i === at ? "text-ink" : "text-ink-tertiary"
            }`}
          >
            {n}
          </span>
        </span>
      ))}
    </nav>
  );
}
