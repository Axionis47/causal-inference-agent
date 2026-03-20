import { Link, useLocation } from 'react-router-dom';

export default function Header() {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="border-b border-ink-200">
      <div className="max-w-4xl mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          <Link to="/" className="font-serif text-lg font-bold text-ink-900 hover:text-accent transition-colors">
            Causal Inference Orchestrator
          </Link>
          <nav className="flex items-center gap-6">
            <Link
              to="/"
              className={`text-sm font-medium transition-colors ${
                isActive('/') ? 'text-accent' : 'text-ink-500 hover:text-ink-900'
              }`}
            >
              New Analysis
            </Link>
            <Link
              to="/jobs"
              className={`text-sm font-medium transition-colors ${
                isActive('/jobs') ? 'text-accent' : 'text-ink-500 hover:text-ink-900'
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
