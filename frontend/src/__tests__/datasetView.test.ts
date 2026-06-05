import { describe, it, expect, beforeEach } from 'vitest';
import { useJobStore } from '../store/jobStore';
import type { DatasetView } from '../services/api';

const fullView = (): DatasetView => ({
  download: {
    status: 'pending',
    url: 'https://www.kaggle.com/datasets/owner/name',
    files: [],
    error: null,
  },
  kaggle_meta: { status: 'pending', data: null, error: null },
  profile: { status: 'pending', data: null, error: null },
  sample: { status: 'pending', files: [] },
});

describe('jobStore datasetView slice', () => {
  beforeEach(() => {
    useJobStore.getState().reset();
  });

  it('starts null', () => {
    expect(useJobStore.getState().datasetView).toBeNull();
  });

  it('setDatasetView replaces the whole view', () => {
    const store = useJobStore.getState();
    store.setDatasetView(fullView());
    const v = useJobStore.getState().datasetView!;
    expect(v.download.status).toBe('pending');
    expect(v.download.url).toContain('kaggle.com');
  });

  it('patchDownload preserves the other blocks', () => {
    const store = useJobStore.getState();
    store.setDatasetView(fullView());
    store.patchDownload({ status: 'downloaded', files: [
      { name: 'a.csv', size_bytes: 1, format: 'csv', used: true },
    ] });
    const v = useJobStore.getState().datasetView!;
    expect(v.download.status).toBe('downloaded');
    expect(v.download.files[0].name).toBe('a.csv');
    expect(v.kaggle_meta.status).toBe('pending');
    expect(v.profile.status).toBe('pending');
  });

  it('patchKaggleMeta from null view bootstraps an empty view', () => {
    // No prior setDatasetView call — patches should still work.
    const store = useJobStore.getState();
    store.patchKaggleMeta({
      status: 'loaded',
      data: {
        description: 'Job-training RCT.',
        column_descriptions: {},
        tags: ['economics'],
        domain: null,
        metadata_quality: 'high',
      },
    });
    const v = useJobStore.getState().datasetView!;
    expect(v.kaggle_meta.status).toBe('loaded');
    expect(v.kaggle_meta.data?.description).toBe('Job-training RCT.');
    expect(v.download.status).toBe('pending');
    expect(v.profile.status).toBe('pending');
  });

  it('patchDownload status=failed carries the error message', () => {
    const store = useJobStore.getState();
    store.patchDownload({ status: 'failed', error: 'Dataset not accessible.' });
    const v = useJobStore.getState().datasetView!;
    expect(v.download.status).toBe('failed');
    expect(v.download.error).toBe('Dataset not accessible.');
  });

  it('patchProfile after metadata-loaded leaves metadata intact', () => {
    const store = useJobStore.getState();
    store.patchKaggleMeta({
      status: 'loaded',
      data: {
        description: null,
        column_descriptions: {},
        tags: [],
        domain: null,
        metadata_quality: 'low',
      },
    });
    store.patchProfile({
      status: 'loaded',
      data: {
        n_samples: 100,
        n_features: 5,
        feature_types: {},
        missing_values: {},
        treatment_candidates: ['t'],
        outcome_candidates: ['y'],
        potential_confounders: [],
        potential_instruments: [],
      },
    });
    const v = useJobStore.getState().datasetView!;
    expect(v.kaggle_meta.status).toBe('loaded');
    expect(v.kaggle_meta.data?.metadata_quality).toBe('low');
    expect(v.profile.status).toBe('loaded');
    expect(v.profile.data?.n_samples).toBe(100);
  });

  it('switching to a different currentJobId clears the datasetView', () => {
    const store = useJobStore.getState();
    store.setDatasetView(fullView());
    store.setCurrentJob({
      id: 'job-A',
      kaggle_url: 'https://kaggle.com/datasets/a/a',
      status: 'pending',
      created_at: '2026-05-10T00:00:00Z',
      updated_at: '2026-05-10T00:00:00Z',
      progress_percentage: 0,
      iteration_count: 0,
    });
    // Re-set the view for job-A so there's something to clear.
    store.setDatasetView(fullView());
    expect(useJobStore.getState().datasetView).not.toBeNull();

    store.setCurrentJob({
      id: 'job-B',
      kaggle_url: 'https://kaggle.com/datasets/b/b',
      status: 'pending',
      created_at: '2026-05-10T00:01:00Z',
      updated_at: '2026-05-10T00:01:00Z',
      progress_percentage: 0,
      iteration_count: 0,
    });
    expect(useJobStore.getState().datasetView).toBeNull();
  });
});
