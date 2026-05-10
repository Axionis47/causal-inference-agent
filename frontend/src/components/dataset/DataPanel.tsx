import { memo, useEffect } from 'react';
import { useDatasetView } from '../../hooks/useDatasetView';
import { useJobStore } from '../../store/jobStore';
import DownloadSection from './DownloadSection';
import KaggleMetaSection from './KaggleMetaSection';
import SchemaSection from './SchemaSection';

interface DataPanelProps {
  jobId: string;
}

function DataPanelImpl({ jobId }: DataPanelProps) {
  const { view } = useDatasetView(jobId);
  const setDatasetView = useJobStore((s) => s.setDatasetView);

  // When the panel unmounts (user navigates away), drop the slice so a
  // future visit to a different job doesn't briefly show stale data.
  useEffect(() => {
    return () => setDatasetView(null);
  }, [setDatasetView]);

  const downloadStatus = view?.download.status ?? 'pending';
  const kaggleStatus = view?.kaggle_meta.status ?? 'pending';
  const profileStatus = view?.profile.status ?? 'pending';

  const allPending =
    downloadStatus === 'pending' &&
    kaggleStatus === 'pending' &&
    profileStatus === 'pending';

  const quality = view?.kaggle_meta.data?.metadata_quality;
  const failureChip =
    downloadStatus === 'failed'
      ? { label: 'download failed', tone: 'red' as const }
      : kaggleStatus === 'error'
        ? { label: 'metadata error', tone: 'red' as const }
        : null;

  return (
    <section className="section">
      <div className="flex items-baseline justify-between mb-4">
        <h2 className="section-title mb-0">Data</h2>
        <div className="flex items-center space-x-2">
          {failureChip && (
            <span className="badge bg-red-50 text-sig-no border border-red-200">
              {failureChip.label}
            </span>
          )}
          {quality && quality !== 'unknown' && (
            <span className="badge bg-ink-50 text-ink-600 border border-ink-200 capitalize">
              quality: {quality}
            </span>
          )}
        </div>
      </div>

      {allPending ? (
        <p className="text-sm text-ink-400 italic">
          <span className="pulse-dot inline-block mr-2">·</span>
          Preparing dataset…
        </p>
      ) : (
        <div className="space-y-0">
          {downloadStatus !== 'pending' && (
            <DownloadSection block={view!.download} />
          )}
          {kaggleStatus !== 'pending' && kaggleStatus !== 'unavailable' && (
            <KaggleMetaSection block={view!.kaggle_meta} />
          )}
          {kaggleStatus === 'unavailable' && (
            <div className="border-t border-ink-100 py-3">
              <p className="text-sm text-ink-400 italic">
                Kaggle did not provide metadata for this dataset.
              </p>
            </div>
          )}
          {profileStatus !== 'pending' && (
            <SchemaSection block={view!.profile} />
          )}
        </div>
      )}
    </section>
  );
}

export default memo(DataPanelImpl);
