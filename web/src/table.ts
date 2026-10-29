// A small table from a CSV an artifact wrote: the header and the first rows, enough to read beside the reply. Pure, so it
// is tested alone; the full file stays one click away.
export interface Table {
  header: string[];
  rows: string[][];
  total: number;
}

export function parseTable(name: string, text: string, limit = 20): Table {
  if (!name.endsWith(".json")) return parseCsv(text, limit);
  const value = JSON.parse(text);
  let header: string[], rows: unknown[][];
  if (Array.isArray(value) && value.every((r) => r !== null && typeof r === "object" && !Array.isArray(r))) {
    header = [...new Set(value.flatMap((r) => Object.keys(r)))];
    rows = value.map((r) => header.map((k) => r[k]));
  } else if (Array.isArray(value.columns) && Array.isArray(value.data)) {
    header = value.columns;
    rows = value.data;
  } else if (value && typeof value === "object" && Object.values(value).every(Array.isArray)) {
    header = Object.keys(value);
    rows = Array.from({ length: header.length ? value[header[0]].length : 0 }, (_, i) => header.map((k) => value[k][i]));
  } else throw new Error("Unsupported JSON table format");
  return {
    header,
    total: rows.length,
    rows: rows.slice(0, limit).map((r) => r.map((v) => (v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v)))),
  };
}

function splitLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (c === '"') quoted = false;
      else cur += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      out.push(cur);
      cur = "";
    } else cur += c;
  }
  out.push(cur);
  return out;
}

export function parseCsv(text: string, limit = 20): Table {
  const lines: string[] = [];
  let quoted = false,
    start = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '"') {
      if (quoted && text[i + 1] === '"') i++;
      else quoted = !quoted;
    } else if (text[i] === "\n" && !quoted) {
      const line = text.slice(start, i).replace(/\r$/, "");
      if (line) lines.push(line);
      start = i + 1;
    }
  }
  if (start < text.length) lines.push(text.slice(start).replace(/\r$/, ""));
  if (!lines.length) return { header: [], rows: [], total: 0 };
  const header = splitLine(lines[0]);
  const body = lines.slice(1);
  return { header, rows: body.slice(0, limit).map(splitLine), total: body.length };
}
