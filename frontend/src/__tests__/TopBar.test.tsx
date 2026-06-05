import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { TopBar } from '../components/job/terminal/TopBar';
import type { JobDetail } from '../services/api';

const job: JobDetail = {
  id: '9a35a319deadbeef',
  kaggle_url: 'https://www.kaggle.com/datasets/owner/name',
  status: 'running',
  created_at: '2026-06-05T12:00:00Z',
  updated_at: '2026-06-05T12:01:00Z',
  iteration_count: 0,
  progress_percentage: 0,
};

function renderTopBar(tokens: number | null) {
  return render(
    <MemoryRouter>
      <TopBar
        job={job}
        agentTones={{}}
        elapsed="00:01:00"
        tokens={tokens}
        isPreview={false}
        onCancel={() => {}}
        cancelPending={false}
      />
    </MemoryRouter>,
  );
}

describe('TopBar', () => {
  it('shows the formatted token total when tokens are present', () => {
    renderTopBar(12345);
    expect(screen.getByText('12,345')).toBeInTheDocument();
  });

  it('falls back to an em dash when there are no tokens yet', () => {
    renderTopBar(null);
    // The label "tokens" is present, and the value is the placeholder dash.
    expect(screen.getByText('tokens')).toBeInTheDocument();
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('treats a zero total as no data (dash, not "0")', () => {
    renderTopBar(0);
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('0')).not.toBeInTheDocument();
  });
});
