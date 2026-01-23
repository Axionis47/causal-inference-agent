import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/common/Header';
import HomePage from './pages/HomePage';
import JobPage from './pages/JobPage';
import JobsListPage from './pages/JobsListPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/jobs" element={<JobsListPage />} />
            <Route path="/jobs/:jobId" element={<JobPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
