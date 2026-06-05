import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { TOAST_DEFAULT_DURATION_MS, TOAST_SUCCESS_DURATION_MS, TOAST_ERROR_DURATION_MS } from './config/constants';
import Header from './components/common/Header';
import ErrorBoundary from './components/common/ErrorBoundary';
import HomePage from './pages/HomePage';
import JobPage from './pages/JobPage';
import JobsListPage from './pages/JobsListPage';

function JournalShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-white">
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
            <Route path="/" element={<JournalShell><HomePage /></JournalShell>} />
            <Route path="/jobs" element={<JournalShell><JobsListPage /></JournalShell>} />
            <Route path="/jobs/:jobId" element={<JobPage />} />
          </Routes>
        </ErrorBoundary>
        <Toaster
          position="top-right"
          toastOptions={{
            duration: TOAST_DEFAULT_DURATION_MS,
            style: {
              borderRadius: '0',
              background: '#1a1a2e',
              color: '#fff',
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
