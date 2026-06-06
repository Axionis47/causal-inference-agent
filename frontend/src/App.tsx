import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { TOAST_DEFAULT_DURATION_MS, TOAST_SUCCESS_DURATION_MS, TOAST_ERROR_DURATION_MS } from './config/constants';
import Header from './components/common/Header';
import ErrorBoundary from './components/common/ErrorBoundary';
import HomePage from './pages/HomePage';
import JobPage from './pages/JobPage';
import JobsListPage from './pages/JobsListPage';

// Shared chrome for the journal-language routes, now migrated to the terminal
// aesthetic (aesthetics.md). The `terminal` class supplies the dark canvas,
// amber focus ring, and cool scrollbar; see index.css.
function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="terminal min-h-screen bg-canvas">
      <Header />
      <main className="max-w-4xl mx-auto px-6 py-10">{children}</main>
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<AppShell><HomePage /></AppShell>} />
            <Route path="/jobs" element={<AppShell><JobsListPage /></AppShell>} />
            <Route path="/jobs/:jobId" element={<JobPage />} />
          </Routes>
        </ErrorBoundary>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: TOAST_DEFAULT_DURATION_MS,
            style: {
              borderRadius: '0',
              background: '#111118',
              color: '#EEEEF0',
              border: '1px solid #2A2A3A',
              fontFamily: 'Inter, system-ui, sans-serif',
              fontSize: '14px',
            },
            success: { duration: TOAST_SUCCESS_DURATION_MS },
            error: { duration: TOAST_ERROR_DURATION_MS },
          }}
        />
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
