import { Link, useLocation } from 'react-router-dom';

export default function Header() {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="bg-canvas-raised border-b border-edge-subtle">
      <div className="max-w-4xl mx-auto px-6">
        <div className="flex items-center justify-between h-12">
          <Link
            to="/"
            className="font-mono text-2xs uppercase tracking-[0.18em] text-ink hover:text-mint transition-colors"
          >
            Causal Inference Orchestrator
          </Link>
          <nav className="flex items-center gap-6">
            <Link
              to="/"
              className={`font-mono text-xs transition-colors ${
                isActive('/') ? 'text-indigo' : 'text-ink-secondary hover:text-ink'
              }`}
            >
              New Analysis
            </Link>
            <Link
              to="/jobs"
              className={`font-mono text-xs transition-colors ${
                isActive('/jobs') ? 'text-indigo' : 'text-ink-secondary hover:text-ink'
              }`}
            >
              Jobs
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
