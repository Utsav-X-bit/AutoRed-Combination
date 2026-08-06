import { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

const RunLoader = lazy(() => import('./pages/RunLoader'));
const InvestigationPage = lazy(() => import('./pages/InvestigationPage'));
const RunComparison = lazy(() => import('./pages/RunComparison'));
const BenchmarkDashboard = lazy(() => import('./pages/BenchmarkDashboard'));

function App() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-slate-500">Loading...</div>}>
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
  );
}

export default App;
