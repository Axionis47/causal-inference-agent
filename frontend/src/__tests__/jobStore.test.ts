import { describe, it, expect, beforeEach } from 'vitest';
import { useJobStore } from '../store/jobStore';

describe('jobStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useJobStore.getState().reset();
  });

  it('should have correct initial state', () => {
    const state = useJobStore.getState();

    expect(state.currentJob).toBeNull();
    expect(state.currentJobId).toBeNull();
    expect(state.results).toBeNull();
    expect(state.traces).toEqual([]);
    expect(state.jobs).toEqual([]);
    expect(state.totalJobs).toBe(0);
    expect(state.isLoading).toBe(false);
    expect(state.isCreating).toBe(false);
    expect(state.error).toBeNull();
  });

  it('should not have startPolling method', () => {
    const state = useJobStore.getState();
    // Polling is handled by React Query, not the store
    expect((state as unknown as Record<string, unknown>).startPolling).toBeUndefined();
  });

  it('should not have stopPolling method', () => {
    const state = useJobStore.getState();
    // Polling is handled by React Query, not the store
    expect((state as unknown as Record<string, unknown>).stopPolling).toBeUndefined();
  });

  it('should set current job correctly', () => {
    const mockJob = {
      id: 'test-job-123',
      kaggle_url: 'https://kaggle.com/datasets/test/data',
      status: 'completed' as const,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      progress_percentage: 100,
      iteration_count: 0,
    };

    useJobStore.getState().setCurrentJob(mockJob);

    const state = useJobStore.getState();
    expect(state.currentJob).toEqual(mockJob);
    expect(state.currentJobId).toBe('test-job-123');
  });

  it('should clear current job when set to null', () => {
    const mockJob = {
      id: 'test-job-123',
      kaggle_url: 'https://kaggle.com/datasets/test/data',
      status: 'completed' as const,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      progress_percentage: 100,
      iteration_count: 0,
    };

    useJobStore.getState().setCurrentJob(mockJob);
    useJobStore.getState().setCurrentJob(null);

    const state = useJobStore.getState();
    expect(state.currentJob).toBeNull();
    expect(state.currentJobId).toBeNull();
  });

  it('should clear error', () => {
    // Manually set an error
    useJobStore.setState({ error: 'Something went wrong' });
    expect(useJobStore.getState().error).toBe('Something went wrong');

    useJobStore.getState().clearError();
    expect(useJobStore.getState().error).toBeNull();
  });

  it('should reset to initial state', () => {
    // Set some state
    useJobStore.setState({
      currentJobId: 'test-123',
      error: 'Some error',
      isLoading: true,
      totalJobs: 5,
    });

    useJobStore.getState().reset();

    const state = useJobStore.getState();
    expect(state.currentJobId).toBeNull();
    expect(state.error).toBeNull();
    expect(state.isLoading).toBe(false);
    expect(state.totalJobs).toBe(0);
  });

  it('should have createJob action defined', () => {
    const state = useJobStore.getState();
    expect(typeof state.createJob).toBe('function');
  });

  it('should have fetchJobs action defined', () => {
    const state = useJobStore.getState();
    expect(typeof state.fetchJobs).toBe('function');
  });

  it('should have cancelJob action defined', () => {
    const state = useJobStore.getState();
    expect(typeof state.cancelJob).toBe('function');
  });

  it('should have fetchResults action defined', () => {
    const state = useJobStore.getState();
    expect(typeof state.fetchResults).toBe('function');
  });

  it('should have fetchTraces action defined', () => {
    const state = useJobStore.getState();
    expect(typeof state.fetchTraces).toBe('function');
  });
});

// Test selectors
describe('jobStore selectors', () => {
  beforeEach(() => {
    useJobStore.getState().reset();
  });

  it('selectIsJobComplete returns true for completed jobs', async () => {
    const { selectIsJobComplete } = await import('../store/jobStore');

    useJobStore.getState().setCurrentJob({
      id: 'test-123',
      kaggle_url: 'https://kaggle.com/datasets/test/data',
      status: 'completed',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      progress_percentage: 100,
      iteration_count: 0,
    });

    expect(selectIsJobComplete(useJobStore.getState())).toBe(true);
  });

  it('selectIsJobFailed returns true for failed jobs', async () => {
    const { selectIsJobFailed } = await import('../store/jobStore');

    useJobStore.getState().setCurrentJob({
      id: 'test-123',
      kaggle_url: 'https://kaggle.com/datasets/test/data',
      status: 'failed',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      progress_percentage: 50,
      iteration_count: 0,
    });

    expect(selectIsJobFailed(useJobStore.getState())).toBe(true);
  });

  it('selectJobProgress returns 0 when no job', async () => {
    const { selectJobProgress } = await import('../store/jobStore');

    expect(selectJobProgress(useJobStore.getState())).toBe(0);
  });
});
