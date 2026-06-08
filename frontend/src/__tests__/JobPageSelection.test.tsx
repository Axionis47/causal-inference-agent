import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import JobPage from '../pages/JobPage';

// Drives the real JobPage in preview mode (no network) to pin the toggle wiring
// that lives in JobPage's onSelect closure: clicking an agent pins it, clicking
// it again unpins it. The unit tests cover the rail and deriveJobView; this
// covers that JobPage threads selection through them.
function renderPreview() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/jobs/__preview']}>
        <Routes>
          <Route path="/jobs/:jobId" element={<JobPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function edaButton() {
  return screen.getByText('eda').closest('button')!;
}

describe('JobPage agent selection toggle', () => {
  it('pins an agent on click and unpins it on a second click', () => {
    renderPreview();

    expect(edaButton().getAttribute('aria-pressed')).toBe('false');

    fireEvent.click(edaButton());
    expect(edaButton().getAttribute('aria-pressed')).toBe('true');

    fireEvent.click(edaButton());
    expect(edaButton().getAttribute('aria-pressed')).toBe('false');
  });
});
