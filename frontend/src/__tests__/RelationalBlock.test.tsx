import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RelationalBlock } from '../components/job/terminal/RelationalBlock';
import type { RelationalProfilePayload } from '../services/api';

const multi: RelationalProfilePayload = {
  shape_hint: 'same_schema_shards',
  files: [
    { file: 'part1.csv', n_rows: 1000, n_cols: 5, key_candidates: ['id'] },
    { file: 'part2.csv', n_rows: 800, n_cols: 5, key_candidates: [] },
  ],
  same_schema_groups: [['part1.csv', 'part2.csv']],
  note: 'One file feeds the analysis; nothing was assembled.',
};

describe('RelationalBlock', () => {
  it('renders the shape, files, and candidate keys', () => {
    render(<RelationalBlock profile={multi} />);
    expect(screen.getByText(/same-schema shards/i)).toBeTruthy();
    expect(screen.getByText('part1.csv')).toBeTruthy();
    expect(screen.getByText('id')).toBeTruthy();
    expect(screen.getByText(/shared schema: part1.csv, part2.csv/)).toBeTruthy();
  });

  it('renders nothing for a single-file bundle', () => {
    const { container } = render(
      <RelationalBlock profile={{ ...multi, files: [multi.files[0]] }} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the profile is null', () => {
    const { container } = render(<RelationalBlock profile={null} />);
    expect(container.firstChild).toBeNull();
  });
});
