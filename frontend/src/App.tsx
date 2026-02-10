import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Header from './components/common/Header';
import ErrorBoundary from './components/common/ErrorBoundary';
import HomePage from './pages/HomePage';
import JobPage from './pages/JobPage';
import JobsListPage from './pages/JobsListPage';

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <Header />
          <main className="container mx-auto px-4 py-8">
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
            duration: 4000,
            style: {
              borderRadius: '8px',
              background: '#333',
              color: '#fff',
            },
            success: { duration: 3000 },
            error: { duration: 5000 },
          }}
        />
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
