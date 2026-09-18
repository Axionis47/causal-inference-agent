// Number and cell formatting for the tables. Facts only, no rounding tricks beyond 4 significant figures.

export function num(v: number | null | undefined, d = 4): string {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "—";
  const s = Number(v).toPrecision(d);
  return s.includes("e") ? s : s.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

export function count(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toLocaleString();
}

export function ci(low: number | null | undefined, high: number | null | undefined): string {
  return `[${num(low)}, ${num(high)}]`;
}

export function cell(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") return num(v);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}
