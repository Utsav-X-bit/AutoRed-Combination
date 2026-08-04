import { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';

const RunLoader = lazy(() => import('./pages/RunLoader'));
const InvestigationPage = lazy(() => import('./pages/InvestigationPage'));
const RunComparison = lazy(() => import('./pages/RunComparison'));
const BenchmarkDashboard = lazy(() => import('./pages/BenchmarkDashboard'));

export default function App() {
  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 dark:bg-stone-950 dark:text-stone-100">
      <Navbar />
      <Suspense
        fallback={
          <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center text-stone-500 dark:text-stone-400">
            <div className="flex items-center gap-3">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-300 border-t-teal-600 dark:border-stone-700 dark:border-t-teal-500" />
              Loading AutoRed…
            </div>
          </div>
        }
      >
        <Routes>
          <Route path="/" element={<Navigate to="/runs" replace />} />
          <Route path="/runs" element={<RunLoader />} />
          <Route path="/run/:runId" element={<InvestigationPage />} />
          <Route path="/compare/:runIdA/:runIdB" element={<RunComparison />} />
          <Route path="/benchmark" element={<Navigate to="/benchmarks" replace />} />
          <Route path="/benchmarks" element={<BenchmarkDashboard />} />
          <Route path="/benchmarks/:benchmarkId" element={<BenchmarkDashboard />} />
        </Routes>
      </Suspense>
    </div>
  );
}
