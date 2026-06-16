import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DagSvg } from '../components/job/terminal/DagSvg';

const props = {
  nodes: ['T', 'Y', 'X', 'U'],
  edges: [
    { source: 'X', target: 'T', edge_type: 'directed' },
    { source: 'T', target: 'Y', edge_type: 'directed' },
  ],
  treatment: 'T',
  outcome: 'Y',
  adjustmentSet: ['X'],
  latent: ['U'],
};

const widthOf = (svg: Element) => Number((svg.getAttribute('viewBox') ?? '').split(' ')[2]);

describe('DagSvg', () => {
  it('draws the nodes and edges of the graph', () => {
    render(<DagSvg {...props} />);
    expect(screen.getByRole('img', { name: 'causal dag' })).toBeTruthy();
    expect(screen.getByText('U')).toBeTruthy();
  });

  it('zooms in by shrinking the viewBox and reset restores it', () => {
    render(<DagSvg {...props} />);
    const svg = screen.getByRole('img', { name: 'causal dag' });
    const start = widthOf(svg);

    fireEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    expect(widthOf(svg)).toBeLessThan(start);

    fireEvent.click(screen.getByRole('button', { name: /reset view/i }));
    expect(widthOf(svg)).toBe(start);
  });

  it('zooms out by widening the viewBox', () => {
    render(<DagSvg {...props} />);
    const svg = screen.getByRole('img', { name: 'causal dag' });
    const start = widthOf(svg);
    fireEvent.click(screen.getByRole('button', { name: /zoom out/i }));
    expect(widthOf(svg)).toBeGreaterThan(start);
  });
});
