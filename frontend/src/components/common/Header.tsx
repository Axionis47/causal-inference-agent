import { Link } from 'react-router-dom';
import { Activity } from 'lucide-react';

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Activity className="w-8 h-8 text-primary-600" />
            <span className="text-xl font-bold text-gray-900">
              Causal Inference Orchestrator
            </span>
          </Link>
          <nav className="flex items-center space-x-4">
            <Link
              to="/"
              className="text-gray-600 hover:text-gray-900 font-medium"
            >
              New Analysis
            </Link>
            <Link
              to="/jobs"
              className="text-gray-600 hover:text-gray-900 font-medium"
            >
              Jobs
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
