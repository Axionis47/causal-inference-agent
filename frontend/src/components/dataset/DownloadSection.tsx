import { memo } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { DownloadBlock } from '../../services/api';
import { useExpandable } from './useExpandable';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

interface DownloadSectionProps {
  block: DownloadBlock;
}

function DownloadSectionImpl({ block }: DownloadSectionProps) {
  const { expanded, toggle } = useExpandable('download');
  const primary = block.files.find((f) => f.used) ?? block.files[0];

  return (
    <div className="border-t border-ink-100 py-3 animate-fade-in">
      {block.status === 'failed' ? (
        <p className="text-sm text-sig-no">
          Download failed{block.error ? ` — ${block.error}` : '.'}
        </p>
      ) : block.status === 'downloading' ? (
        <p className="text-sm text-ink-500">
          <span className="pulse-dot inline-block mr-2">·</span>
          Downloading from {hostnameOf(block.url)}…
        </p>
      ) : (
        <>
          <button
            type="button"
            onClick={toggle}
            className="flex w-full items-center text-left text-sm text-ink-700 hover:text-ink-900"
            aria-expanded={expanded}
          >
            {expanded ? (
              <ChevronDown className="w-4 h-4 mr-2 text-ink-400" />
            ) : (
              <ChevronRight className="w-4 h-4 mr-2 text-ink-400" />
            )}
            <span>
              Downloaded
              {primary ? ` · ${primary.name}` : ''}
              {primary ? ` · ${formatBytes(primary.size_bytes)}` : ''}
              {block.files.length > 1 ? ` · +${block.files.length - 1} more` : ''}
            </span>
          </button>
          {expanded && (
            <div className="mt-3 ml-6">
              {block.url && (
                <p className="text-xs text-ink-400 font-mono break-all mb-2">
                  {block.url}
                </p>
              )}
              <table className="journal-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Format</th>
                    <th>Size</th>
                    <th>Used</th>
                  </tr>
                </thead>
                <tbody>
                  {block.files.map((f) => (
                    <tr key={f.name}>
                      <td className="data-value">{f.name}</td>
                      <td className="text-xs uppercase tracking-wider text-ink-500">
                        {f.format}
                      </td>
                      <td className="data-value text-right">
                        {formatBytes(f.size_bytes)}
                      </td>
                      <td>
                        {f.used ? (
                          <span className="text-xs text-ink-700">yes</span>
                        ) : (
                          <span className="text-xs text-ink-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function hostnameOf(url: string | null): string {
  if (!url) return 'source';
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'source';
  }
}

export default memo(DownloadSectionImpl);
