import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import HomePage from '../pages/HomePage';

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    createJob: vi.fn(),
  };
});

import { createJob } from '../services/api';

function renderPage() {
  vi.mocked(createJob).mockResolvedValue({
    id: 'abc12345',
    kaggle_url: 'https://www.kaggle.com/datasets/owner/name',
    status: 'pending',
    created_at: '2026-05-10T12:00:00Z',
    updated_at: '2026-05-10T12:00:00Z',
  });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

function fillUrlAndSubmit() {
  fireEvent.change(screen.getByLabelText(/Dataset URL/i), {
    target: { value: 'https://www.kaggle.com/datasets/owner/name' },
  });
  // Treatment and outcome are required, so a valid submission must set them.
  fireEvent.change(screen.getByLabelText(/Treatment variable/i), {
    target: { value: 'treat' },
  });
  fireEvent.change(screen.getByLabelText(/Outcome variable/i), {
    target: { value: 're78' },
  });
  fireEvent.click(screen.getByRole('button', { name: /Run Causal Analysis/i }));
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('HomePage orchestrator picker', () => {
  it('hides the orchestrator picker by default', () => {
    renderPage();
    expect(screen.queryByText('Orchestrator')).toBeNull();
    expect(screen.getByText(/Advanced options/i)).toBeTruthy();
  });

  it('reveals the picker when Advanced options is clicked', () => {
    renderPage();
    fireEvent.click(screen.getByText(/Advanced options/i));
    expect(screen.getByText('Orchestrator')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Standard' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'ReAct' })).toBeTruthy();
    expect(
      screen.getByText(/Fixed pipeline of 13 agents/i),
    ).toBeTruthy();
  });

  it('caption swaps when ReAct is selected', () => {
    renderPage();
    fireEvent.click(screen.getByText(/Advanced options/i));
    fireEvent.click(screen.getByRole('button', { name: 'ReAct' }));
    expect(
      screen.getByText(/Orchestrator decides which agent runs next/i),
    ).toBeTruthy();
  });

  it("submits orchestrator_mode: 'standard' by default", async () => {
    renderPage();
    fillUrlAndSubmit();
    await waitFor(() =>
      expect(vi.mocked(createJob)).toHaveBeenCalled(),
    );
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload.orchestrator_mode).toBe('standard');
  });

  it("submits orchestrator_mode: 'react' after selecting ReAct", async () => {
    renderPage();
    fireEvent.click(screen.getByText(/Advanced options/i));
    fireEvent.click(screen.getByRole('button', { name: 'ReAct' }));
    fillUrlAndSubmit();
    await waitFor(() =>
      expect(vi.mocked(createJob)).toHaveBeenCalled(),
    );
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload.orchestrator_mode).toBe('react');
  });
});

describe('HomePage required treatment/outcome', () => {
  it('blocks submit and shows an error when treatment/outcome are empty', () => {
    renderPage();
    fireEvent.change(screen.getByLabelText(/Dataset URL/i), {
      target: { value: 'https://www.kaggle.com/datasets/owner/name' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: /Run Causal Analysis/i }),
    );
    expect(screen.getByText(/Treatment variable is required/i)).toBeTruthy();
    expect(vi.mocked(createJob)).not.toHaveBeenCalled();
  });

  it('forwards treatment, outcome and optional context in the payload', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText(/Dataset URL/i), {
      target: { value: 'https://www.kaggle.com/datasets/owner/name' },
    });
    fireEvent.change(screen.getByLabelText(/Treatment variable/i), {
      target: { value: 'treat' },
    });
    fireEvent.change(screen.getByLabelText(/Outcome variable/i), {
      target: { value: 're78' },
    });
    fireEvent.change(screen.getByLabelText(/Context/i), {
      target: { value: 'NSW job-training program; randomized treated arm' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: /Run Causal Analysis/i }),
    );
    await waitFor(() => expect(vi.mocked(createJob)).toHaveBeenCalled());
    const payload = vi.mocked(createJob).mock.calls[0][0];
    expect(payload.treatment_variable).toBe('treat');
    expect(payload.outcome_variable).toBe('re78');
    expect(payload.user_context).toBe(
      'NSW job-training program; randomized treated arm',
    );
  });
});
