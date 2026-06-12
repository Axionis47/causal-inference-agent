// Presentable view of a table artifact: parse the CSV bytes and render a
// real grid. Long tables are capped; the cap is stated, never silent.

import { parseCsv } from '../csv';

const MAX_ROWS = 50;

export function TableView({ csv }: { csv: string }) {
  const rows = parseCsv(csv);
  if (rows.length === 0) {
    return <p className="text-2xs font-mono text-ink-tertiary">empty table</p>;
  }
  const [header, ...body] = rows;
  const shown = body.slice(0, MAX_ROWS);

  return (
    <div className="overflow-x-auto">
      <table className="font-mono text-2xs tabular border-collapse">
        <thead>
          <tr>
            {header.map((h, i) => (
              <th
                key={i}
                className="text-left text-ink-tertiary font-normal uppercase tracking-wider px-2 py-1 border-b border-edge-subtle whitespace-nowrap"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i} className="border-b border-edge-subtle/30">
              {r.map((cell, j) => (
                <td key={j} className="px-2 py-0.5 text-ink-secondary whitespace-nowrap">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {body.length > MAX_ROWS && (
        <p className="text-2xs font-mono text-ink-tertiary mt-1">
          first {MAX_ROWS} of {body.length} rows; open raw for the full table
        </p>
      )}
    </div>
  );
}
