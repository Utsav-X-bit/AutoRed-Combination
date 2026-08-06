import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import type { AutoRedRun } from '../types/autored';
import { normalizeRun } from '../utils/normalizeRun';

export default function RunComparison() {
  const { runIdA, runIdB } = useParams<{ runIdA: string; runIdB: string }>();
  const navigate = useNavigate();
  const [runA, setRunA] = useState<AutoRedRun | null>(null);
  const [runB, setRunB] = useState<AutoRedRun | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runIdA || !runIdB) return;
    Promise.all([
      fetch(`/api/run/${encodeURIComponent(runIdA)}`).then(r => {
        if (!r.ok) throw new Error(`Run ${runIdA}: ${r.status}`);
        return r.json();
      }),
      fetch(`/api/run/${encodeURIComponent(runIdB)}`).then(r => {
        if (!r.ok) throw new Error(`Run ${runIdB}: ${r.status}`);
        return r.json();
      }),
    ]).then(([a, b]) => {
      setRunA(normalizeRun(a));
      setRunB(normalizeRun(b));
    }).catch(err => {
      console.error('Failed to load runs:', err);
      navigate('/runs');
    }).finally(() => setLoading(false));
  }, [runIdA, runIdB]);

  if (loading) return <div className="p-8 text-center text-slate-500">Loading runs...</div>;
  if (!runA || !runB) return null;

  // Compute strategy comparison
  const strategiesA = Object.keys(runA.strategy_stats);
  const strategiesB = Object.keys(runB.strategy_stats);
  const allStrategies = [...new Set([...strategiesA, ...strategiesB])];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/runs')} className="text-sm text-slate-500 hover:text-slate-900">
              ← Runs
            </button>
            <span className="text-slate-300">|</span>
            <h1 className="text-xl font-bold text-slate-900">Run Comparison</h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 gap-6">
          {/* Run A */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h2 className="text-sm font-bold text-slate-900 mb-3">Run A</h2>
            <p className="font-mono text-xs text-slate-500 mb-3">{runA.experiment.run_id}</p>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-xs text-slate-500">Success</p>
                <p className={`font-bold ${runA.result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
                  {runA.result.ground_truth_success ? '✓' : '✗'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Attempts</p>
                <p className="font-bold">{runA.result.total_attempts}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Generator</p>
                <p className="font-medium truncate">{runA.models.generator.name.split('/').pop()}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Victim</p>
                <p className="font-medium truncate">{runA.models.victim.name.split('/').pop()}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Extractor</p>
                <p className={`font-bold ${runA.result.extractor_success ? 'text-green-600' : 'text-red-600'}`}>
                  {runA.result.extractor_success ? '✓' : '✗'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Run Time</p>
                <p className="font-medium">{runA.timing.total_run_time.toFixed(1)}s</p>
              </div>
            </div>
          </div>

          {/* Run B */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h2 className="text-sm font-bold text-slate-900 mb-3">Run B</h2>
            <p className="font-mono text-xs text-slate-500 mb-3">{runB.experiment.run_id}</p>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-xs text-slate-500">Success</p>
                <p className={`font-bold ${runB.result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
                  {runB.result.ground_truth_success ? '✓' : '✗'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Attempts</p>
                <p className="font-bold">{runB.result.total_attempts}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Generator</p>
                <p className="font-medium truncate">{runB.models.generator.name.split('/').pop()}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Victim</p>
                <p className="font-medium truncate">{runB.models.victim.name.split('/').pop()}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Extractor</p>
                <p className={`font-bold ${runB.result.extractor_success ? 'text-green-600' : 'text-red-600'}`}>
                  {runB.result.extractor_success ? '✓' : '✗'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Run Time</p>
                <p className="font-medium">{runB.timing.total_run_time.toFixed(1)}s</p>
              </div>
            </div>
          </div>
        </div>

        {/* Strategy Comparison */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Strategy</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Run A Successes</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Run A Failures</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Run B Successes</th>
                  <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Run B Failures</th>
                </tr>
              </thead>
              <tbody>
                {allStrategies.map(s => {
                  const aStat = runA.strategy_stats[s] || { successes: 0, partial_leaks: 0, failures: 0, total_score: 0 };
                  const bStat = runB.strategy_stats[s] || { successes: 0, partial_leaks: 0, failures: 0, total_score: 0 };
                  return (
                    <tr key={s} className="border-b border-slate-100">
                      <td className="py-2 px-3 font-mono text-xs">{s}</td>
                      <td className="py-2 px-3 text-center text-green-600 font-bold">{aStat.successes}</td>
                      <td className="py-2 px-3 text-center text-red-600 font-bold">{aStat.failures}</td>
                      <td className="py-2 px-3 text-center text-green-600 font-bold">{bStat.successes}</td>
                      <td className="py-2 px-3 text-center text-red-600 font-bold">{bStat.failures}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Per-Attempt Comparison */}
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-bold text-slate-900 mb-3">Per-Attempt Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">#</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Run A Strategy</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Run A Result</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Run B Strategy</th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Run B Result</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: Math.max(runA.attempts.length, runB.attempts.length) }).map((_, i) => {
                  const a = runA.attempts[i];
                  const b = runB.attempts[i];
                  return (
                    <tr key={i} className="border-b border-slate-100">
                      <td className="py-2 px-3 text-slate-400">{i + 1}</td>
                      <td className="py-2 px-3">
                        {a ? (
                          <>
                            <span className="font-mono text-xs">{a.generator.strategy}</span>
                            {a.ground_truth_found && <span className="ml-1 text-green-600">✓</span>}
                          </>
                        ) : <span className="text-slate-300">—</span>}
                      </td>
                      <td className="py-2 px-3">
                        {a ? (
                          <span className={`text-xs ${a.extractor_match ? 'text-green-600' : 'text-red-600'}`}>
                            {a.extractor_match ? '✓' : '✗'}
                          </span>
                        ) : <span className="text-slate-300">—</span>}
                      </td>
                      <td className="py-2 px-3">
                        {b ? (
                          <>
                            <span className="font-mono text-xs">{b.generator.strategy}</span>
                            {b.ground_truth_found && <span className="ml-1 text-green-600">✓</span>}
                          </>
                        ) : <span className="text-slate-300">—</span>}
                      </td>
                      <td className="py-2 px-3">
                        {b ? (
                          <span className={`text-xs ${b.extractor_match ? 'text-green-600' : 'text-red-600'}`}>
                            {b.extractor_match ? '✓' : '✗'}
                          </span>
                        ) : <span className="text-slate-300">—</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex gap-3">
          <button
            onClick={() => navigate(`/run/${runA.experiment.run_id}`)}
            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium transition-colors"
          >
            View Run A Detail
          </button>
          <button
            onClick={() => navigate(`/run/${runB.experiment.run_id}`)}
            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium transition-colors"
          >
            View Run B Detail
          </button>
        </div>
      </main>
    </div>
  );
}
