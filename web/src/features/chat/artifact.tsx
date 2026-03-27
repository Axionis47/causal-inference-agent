import { Button } from "@/components/ui/button";
import type { components } from "@/lib/api/schema";

type Artifact = components["schemas"]["ArtifactView"];

export function ArtifactCard({
  artifact,
  conversationId,
  selected,
  onSelect,
}: {
  artifact: Artifact;
  conversationId: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const base = `/api/conversations/${conversationId}/artifacts/${artifact.id}/files/`;
  return (
    <section className="chat-artifact" aria-label="Visualization result">
      <div className="flex flex-wrap justify-between gap-3 items-center">
        <h3 className="panel-title">Exploration result</h3>
        <Button
          variant={selected ? "secondary" : "outline"}
          size="sm"
          onClick={onSelect}
          aria-pressed={selected}
        >
          {selected ? "Selected for follow-up" : "Use in follow-up"}
        </Button>
      </div>
      {artifact.record.figure && (
        <a
          href={`${base}figure.png`}
          target="_blank"
          rel="noreferrer"
          aria-label="Open full visualization"
        >
          <img
            className="chat-figure"
            src={`${base}figure.png`}
            alt={artifact.record.summary}
            loading="lazy"
          />
        </a>
      )}
      <p className="mt-3 whitespace-pre-wrap">{artifact.record.summary}</p>
      <p className="subtext mt-3">{artifact.record.method}</p>
      <ul className="chat-findings">
        {(artifact.record.findings ?? []).map((finding, index) => (
          <li key={index}>{finding}</li>
        ))}
      </ul>
      <div className="chat-limitations">
        <strong>Limitations</strong>
        <ul>
          {(artifact.record.limitations ?? []).map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>
      <details className="mt-4">
        <summary className="cursor-pointer">Computed facts & files</summary>
        <div
          className="table-scroll mt-3"
          tabIndex={0}
          role="region"
          aria-label="Computed facts"
        >
          <table className="dataset-table">
            <caption className="sr-only">Recorded numerical evidence</caption>
            <thead>
              <tr>
                <th scope="col">Fact</th>
                <th scope="col">Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(artifact.facts).map(([key, value]) => (
                <tr key={key}>
                  <th scope="row">{key}</th>
                  <td>
                    {value.toLocaleString(undefined, {
                      maximumSignificantDigits: 10,
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap gap-3 mt-4">
          {artifact.filenames.map((name) => (
            <a
              className="context-link"
              key={name}
              href={base + encodeURIComponent(name)}
              download
            >
              {name}
            </a>
          ))}
        </div>
      </details>
    </section>
  );
}
