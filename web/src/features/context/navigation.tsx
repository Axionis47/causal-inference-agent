import { Link, useNavigate } from "@tanstack/react-router";
import { label } from "./types";
import type { Workspace } from "./types";

const route = "/datasets/$datasetId/context/$versionId" as const;

export function ContextNavigation({
  datasetId,
  versionId,
  revision,
  section,
  columns,
  onNavigate,
}: {
  datasetId: string;
  versionId: string;
  revision: string;
  section: string;
  columns: Workspace["columns"];
  onNavigate: () => void;
}) {
  const navigate = useNavigate();
  const isColumn = section.startsWith("columns.");
  const columnSection = isColumn
    ? section
    : columns[0] && `columns.${columns[0].column_id}`;
  return (
    <>
      <nav className="context-nav" aria-label="Context sections">
        {["dataset", "columns", "concepts", "relationships"].map((key) => {
          if (key === "columns" && !columnSection) return null;
          const target = key === "columns" ? columnSection! : key;
          const active = key === "columns" ? isColumn : section === key;
          return (
            <Link
              key={key}
              onClick={onNavigate}
              to={route}
              params={{ datasetId, versionId }}
              search={{ revision, section: target }}
              className={active ? "active" : ""}
              aria-current={active ? "page" : undefined}
            >
              {label(key)}
            </Link>
          );
        })}
      </nav>
      {isColumn && (
        <div className="context-column-selector">
          <label htmlFor="context-column" className="eyebrow">
            Column
          </label>
          <select
            id="context-column"
            name="column"
            className="context-input"
            autoComplete="off"
            value={section}
            onChange={(event) => {
              onNavigate();
              void navigate({
                to: route,
                params: { datasetId, versionId },
                search: { revision, section: event.target.value },
              });
            }}
          >
            {columns.map((column) => (
              <option key={column.column_id} value={`columns.${column.column_id}`}>
                {column.name}
              </option>
            ))}
          </select>
        </div>
      )}
    </>
  );
}
