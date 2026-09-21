import type { ClaimView, StatusView } from "../types";

// One mark per claim kind, in the table's order. Per-column claims fold into one mark.
const RANK: Record<string, number> = { confirmed: 0, unknown: 1, drafted: 2, empty: 3, refuted: 4, contradiction: 4 };

function worst(statuses: string[]): string {
  if (!statuses.length) return "idle";
  let w = statuses[0];
  for (const s of statuses) if ((RANK[s] ?? 3) > (RANK[w] ?? 3)) w = s;
  return w === "contradiction" ? "refuted" : w;
}

export default function StatusStrip({ claims, status, onOpen }: { claims: ClaimView[]; status: StatusView | null; onOpen?: () => void }) {
  if (!claims.length) return null;
  const required = new Set(status?.required ?? []);
  const kinds: string[] = [];
  const by: Record<string, ClaimView[]> = {};
  for (const c of claims) {
    if (!by[c.kind]) {
      by[c.kind] = [];
      kinds.push(c.kind);
    }
    by[c.kind].push(c);
  }
  const marks = kinds.map((k) => {
    const cs = by[k];
    const req = cs.filter((c) => required.has(c.key));
    const pool = req.length ? req : cs;
    const state = required.size && !req.length ? "idle" : worst(pool.map((c) => c.status));
    const settled = pool.filter((c) => c.status === "confirmed" || c.status === "unknown" || c.status === "contradiction").length;
    const label = cs.length > 1 ? `${k} ${settled}/${pool.length}` : k.replace(/_/g, " ");
    const tip = cs.map((c) => `${c.key}: ${c.status}`).join("\n");
    return (
      <button type="button" className={`mark ${state}`} key={k} title={tip} onClick={onOpen}>
        <span className="dot" />
        <span className="lbl">{label}</span>
      </button>
    );
  });
  const settled = status?.settled.length ?? 0;
  const total = (status?.settled.length ?? 0) + (status?.open.length ?? 0);
  const struck = Object.entries(status?.struck ?? {});
  return (
    <div className="strip" aria-label="Claims settled so far">
      <div className="marks">{marks}</div>
      {status && (
        <div className="line">
          <b>
            settled {settled}/{total}
          </b>
          {status.open.length > 0 && (
            <>
              {" "}
              · open: {status.open.slice(0, 6).join(", ")}
              {status.open.length > 6 ? ` +${status.open.length - 6} more` : ""}
            </>
          )}
          {" · "}
          {status.surviving.length ? <>in play: {status.surviving.join(", ")}</> : "no design fits yet"}
          {" · "}
          {status.ready ? <b>ready</b> : "not ready"}
        </div>
      )}
      {struck.length > 0 && <div className="struck">struck out: {struck.map(([f, w]) => `${f} (${w})`).join("; ")}</div>}
    </div>
  );
}
