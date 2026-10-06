// A small table from a CSV an artifact wrote: the header and the first rows, enough to read beside the reply. Pure, so it
// is tested alone; the full file stays one click away.
export interface Table {
  header: string[];
  rows: string[][];
  total: number;
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
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (!lines.length) return { header: [], rows: [], total: 0 };
  const header = splitLine(lines[0]);
  const body = lines.slice(1);
  return { header, rows: body.slice(0, limit).map(splitLine), total: body.length };
}
