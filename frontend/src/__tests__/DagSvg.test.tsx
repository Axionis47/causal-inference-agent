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
  it('renders the graph and groups confounders into latent and measured', () => {
    render(<DagSvg {...props} />);
    expect(screen.getByRole('img', { name: 'causal dag' })).toBeTruthy();
    // The left column carries a header per group: U is latent, X is measured.
    expect(screen.getByText('unmeasured')).toBeTruthy();
    expect(screen.getByText('measured')).toBeTruthy();
  });

  it('truncates a long label but keeps the full name on hover', () => {
    const longName = 'Unmeasured Socioeconomic Status/Family Background';
    render(
      <DagSvg
        nodes={['T', 'Y', longName]}
        edges={[
          { source: longName, target: 'T', edge_type: 'directed' },
          { source: 'T', target: 'Y', edge_type: 'directed' },
        ]}
        treatment="T"
        outcome="Y"
        adjustmentSet={[]}
        latent={[longName]}
      />,
    );
    // Full name lives only in the <title> (hover); the visible label is cut with
    // an ellipsis so it cannot overrun its neighbours.
    expect(screen.getByText(longName, { selector: 'title' })).toBeTruthy();
    expect(screen.queryByText(longName, { selector: 'text' })).toBeNull();
    expect(screen.getByText(/…$/)).toBeTruthy();
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
