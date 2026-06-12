// Presentable view of a markdown artifact (summaries, stories, the final
// report). Element styling is mapped explicitly to the terminal theme; there
// is no global prose stylesheet.

import ReactMarkdown, { type Components } from 'react-markdown';

const components: Components = {
  h1: (props) => <h3 className="text-sm font-semibold text-ink mt-3 mb-1" {...props} />,
  h2: (props) => <h4 className="text-xs font-semibold text-ink mt-2.5 mb-1" {...props} />,
  h3: (props) => (
    <h5
      className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary mt-2 mb-1"
      {...props}
    />
  ),
  p: (props) => <p className="text-xs text-ink-secondary leading-relaxed mb-1.5" {...props} />,
  ul: (props) => <ul className="list-disc pl-4 mb-1.5 space-y-0.5" {...props} />,
  ol: (props) => <ol className="list-decimal pl-4 mb-1.5 space-y-0.5" {...props} />,
  li: (props) => <li className="text-xs text-ink-secondary" {...props} />,
  strong: (props) => <strong className="text-ink font-semibold" {...props} />,
  em: (props) => <em className="text-ink-secondary" {...props} />,
  code: (props) => (
    <code className="font-mono text-2xs text-indigo bg-canvas px-1 py-0.5" {...props} />
  ),
  pre: (props) => (
    <pre
      className="font-mono text-2xs text-ink-secondary bg-canvas border border-edge-subtle p-2 overflow-x-auto mb-1.5"
      {...props}
    />
  ),
  a: (props) => (
    <a
      className="text-indigo underline underline-offset-2"
      target="_blank"
      rel="noreferrer"
      {...props}
    />
  ),
  blockquote: (props) => (
    <blockquote className="border-l-2 border-edge-subtle pl-2 text-ink-tertiary" {...props} />
  ),
};

export function MarkdownView({ text }: { text: string }) {
  return (
    <div className="min-w-0">
      <ReactMarkdown components={components}>{text}</ReactMarkdown>
    </div>
  );
}
