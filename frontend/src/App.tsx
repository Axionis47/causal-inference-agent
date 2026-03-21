import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { TOAST_DEFAULT_DURATION_MS, TOAST_SUCCESS_DURATION_MS, TOAST_ERROR_DURATION_MS } from './config/constants';
import Header from './components/common/Header';
import ErrorBoundary from './components/common/ErrorBoundary';
import HomePage from './pages/HomePage';
import JobPage from './pages/JobPage';
import JobsListPage from './pages/JobsListPage';

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="min-h-screen bg-white">
          <Header />
          <main className="max-w-4xl mx-auto px-6 py-10">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/jobs" element={<JobsListPage />} />
                <Route path="/jobs/:jobId" element={<JobPage />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>
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
