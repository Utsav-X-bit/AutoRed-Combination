import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type {
  BenchmarkDetail,
  BenchmarkListItem,
  RunListItem,
  TraceArchiveDetail,
  TraceRunListItem,
} from '../types/autored';
import { normalizeRunList } from '../utils/normalizeRun';

const DIRECT_PAGE_SIZE = 40;

const asDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatDateTime = (value: string) => {
  const date = asDate(value);
  return date ? date.toLocaleString() : 'n/a';
};

const formatPct = (value: number | undefined | null) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return '0.0%';
  return `${(number * 100).toFixed(1)}%`;
};

const sortRuns = (items: RunListItem[]) =>
  [...items].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return b.run_id.localeCompare(a.run_id);
  });

const sortBenchmarks = (items: BenchmarkListItem[]) =>
  [...items].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return b.benchmark_id.localeCompare(a.benchmark_id);
  });

const sortTraceArchives = (items: TraceArchiveDetail[]) =>
  [...items].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return b.archive_id.localeCompare(a.archive_id);
  });

const sortTraceRuns = (items: TraceRunListItem[]) =>
  [...items].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return b.run_id.localeCompare(a.run_id);
  });

type BenchmarkState = {
  expanded: boolean;
  loading: boolean;
  error: string | null;
  detail: BenchmarkDetail | null;
  expandedArchives: Record<string, boolean>;
  archiveRunLimit: Record<string, number>;
};

export default function RunLoader() {
  const navigate = useNavigate();
  const [directRuns, setDirectRuns] = useState<RunListItem[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkListItem[]>([]);
  const [benchmarkStates, setBenchmarkStates] = useState<Record<string, BenchmarkState>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runQuery, setRunQuery] = useState('');
  const [benchmarkQuery, setBenchmarkQuery] = useState('');
  const [directVisible, setDirectVisible] = useState(DIRECT_PAGE_SIZE);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const [runRes, benchRes] = await Promise.all([
          fetch(`/api/runs?limit=${DIRECT_PAGE_SIZE}&offset=0`),
          fetch('/api/benchmarks'),
        ]);

        if (!runRes.ok) throw new Error(`Runs endpoint returned ${runRes.status}`);
        if (!benchRes.ok) throw new Error(`Benchmarks endpoint returned ${benchRes.status}`);

        const [runsJson, benchmarksJson] = await Promise.all([runRes.json(), benchRes.json()]);
        if (cancelled) return;

        setDirectRuns(sortRuns(normalizeRunList(runsJson)));
        setBenchmarks(sortBenchmarks(benchmarksJson as BenchmarkListItem[]));
        setBenchmarkStates(
          Object.fromEntries(
            (benchmarksJson as BenchmarkListItem[]).map((item) => [item.benchmark_id, {
              expanded: false,
              loading: false,
              error: null,
              detail: null,
              expandedArchives: {},
              archiveRunLimit: {},
            }]),
          ),
        );
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load run history');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredDirectRuns = useMemo(() => {
    const query = runQuery.trim().toLowerCase();
    const items = directRuns.filter((run) => {
      if (!query) return true;
      return [
        run.run_id,
        run.scenario_id,
        run.access_code,
        run.generator,
        run.victim,
        run.timestamp,
      ].some((field) => field.toLowerCase().includes(query));
    });
    return items;
  }, [directRuns, runQuery]);

  const filteredBenchmarks = useMemo(() => {
    const query = benchmarkQuery.trim().toLowerCase();
    return benchmarks.filter((benchmark) => {
      if (!query) return true;
      return [
        benchmark.benchmark_id,
        benchmark.timestamp,
        benchmark.file_path,
        ...(benchmark.trace_archives ?? []),
      ].some((field) => field.toLowerCase().includes(query));
    });
  }, [benchmarkQuery, benchmarks]);

  const updateBenchmarkState = (benchmarkId: string, updater: (state: BenchmarkState) => BenchmarkState) => {
    setBenchmarkStates((current) => {
      const next = current[benchmarkId];
      if (!next) return current;
      return { ...current, [benchmarkId]: updater(next) };
    });
  };

  const expandBenchmark = async (benchmarkId: string) => {
    const current = benchmarkStates[benchmarkId];
    if (!current) return;
    if (current.expanded) {
      updateBenchmarkState(benchmarkId, (state) => ({ ...state, expanded: false }));
      return;
    }

    updateBenchmarkState(benchmarkId, (state) => ({ ...state, expanded: true }));
    if (current.detail || current.loading) return;

    updateBenchmarkState(benchmarkId, (state) => ({ ...state, loading: true, error: null }));
    try {
      const res = await fetch(`/api/benchmarks/${encodeURIComponent(benchmarkId)}`);
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const detail = (await res.json()) as BenchmarkDetail;
      updateBenchmarkState(benchmarkId, (state) => ({
        ...state,
        loading: false,
        detail,
        archiveRunLimit: Object.fromEntries(
          (detail.trace_archives ?? []).map((archive) => [archive.archive_id, 20]),
        ),
        expandedArchives: Object.fromEntries(
          (detail.trace_archives ?? []).map((archive) => [archive.archive_id, false]),
        ),
      }));
    } catch (err) {
      updateBenchmarkState(benchmarkId, (state) => ({
        ...state,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load benchmark detail',
      }));
    }
  };

  const toggleArchive = (benchmarkId: string, archiveId: string) => {
    updateBenchmarkState(benchmarkId, (state) => ({
      ...state,
      expandedArchives: {
        ...state.expandedArchives,
        [archiveId]: !state.expandedArchives[archiveId],
      },
    }));
  };

  const loadMoreArchiveRuns = (benchmarkId: string, archiveId: string) => {
    updateBenchmarkState(benchmarkId, (state) => ({
      ...state,
      archiveRunLimit: {
        ...state.archiveRunLimit,
        [archiveId]: (state.archiveRunLimit[archiveId] ?? 20) + 20,
      },
    }));
  };

  const loadMoreDirectRuns = () => {
    setDirectVisible((current) => current + DIRECT_PAGE_SIZE);
  };

  const visibleDirectRuns = filteredDirectRuns.slice(0, directVisible);
  const hasMoreDirectRuns = filteredDirectRuns.length > visibleDirectRuns.length;

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-slate-500">Loading run explorer...</div>;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center px-6">
        <div className="max-w-xl w-full bg-white border border-slate-200 rounded-xl p-6">
          <h1 className="text-lg font-semibold text-slate-900">Run explorer unavailable</h1>
          <p className="mt-2 text-sm text-slate-600">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 text-sm font-medium bg-slate-900 text-white rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-slate-900">Run Explorer</h1>
            <p className="text-sm text-slate-500">
              Benchmark folders, dated trace runs, and direct results from `results/`.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => navigate('/benchmarks')}
              className="px-4 py-2 text-sm font-medium bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200"
            >
              Benchmarks
            </button>
            <button
              onClick={() => navigate('/benchmark')}
              className="px-4 py-2 text-sm font-medium bg-slate-900 text-white rounded-lg hover:bg-slate-800"
            >
              Open Benchmark Dashboard
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-8">
        <section className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Benchmark folders</h2>
              <p className="text-xs text-slate-500">
                Expand a benchmark to load its dated trace archives and run rows on demand.
              </p>
            </div>
            <input
              value={benchmarkQuery}
              onChange={(e) => setBenchmarkQuery(e.target.value)}
              placeholder="Filter benchmarks"
              className="w-full sm:w-72 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-slate-400"
            />
          </div>

          <div className="space-y-3">
            {filteredBenchmarks.map((benchmark) => {
              const state = benchmarkStates[benchmark.benchmark_id];
              if (!state) return null;
              const summary = state.detail?.summary ?? benchmark;
              const archives = sortTraceArchives(state.detail?.trace_archives ?? []);

              return (
                <div key={benchmark.benchmark_id} className="border border-slate-200 rounded-lg overflow-hidden">
                  <button
                    onClick={() => expandBenchmark(benchmark.benchmark_id)}
                    className="w-full flex items-center justify-between gap-4 px-4 py-3 text-left hover:bg-slate-50"
                  >
                    <div className="min-w-0">
                      <p className="font-mono text-xs text-slate-500 truncate">{benchmark.benchmark_id}</p>
                      <p className="text-sm text-slate-700">
                        {formatDateTime(benchmark.timestamp)} · {benchmark.trace_archive_count} archive(s)
                      </p>
                    </div>
                    <div className="text-right text-xs text-slate-500">
                      <div>{formatPct(summary.success_rate)}</div>
                      <div>{summary.total_rounds ?? benchmark.total_rounds ?? 0} rounds</div>
                    </div>
                  </button>

                  {state.expanded && (
                    <div className="border-t border-slate-200 bg-slate-50 p-4 space-y-4">
                      {state.loading && <p className="text-sm text-slate-500">Loading benchmark detail...</p>}
                      {state.error && <p className="text-sm text-red-600">{state.error}</p>}
                      {state.detail && (
                        <>
                          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 text-sm">
                            <div className="bg-white border border-slate-200 rounded-lg p-3">
                              <p className="text-xs text-slate-500">Success rate</p>
                              <p className="font-semibold text-slate-900">{formatPct(state.detail.summary.success_rate ?? benchmark.success_rate)}</p>
                            </div>
                            <div className="bg-white border border-slate-200 rounded-lg p-3">
                              <p className="text-xs text-slate-500">Verified success</p>
                              <p className="font-semibold text-slate-900">
                                {state.detail.summary.verified_success ?? benchmark.verified_success ?? 0}
                              </p>
                            </div>
                            <div className="bg-white border border-slate-200 rounded-lg p-3">
                              <p className="text-xs text-slate-500">Avg attempts</p>
                              <p className="font-semibold text-slate-900">
                                {Number(state.detail.summary.avg_attempts_on_success ?? benchmark.avg_attempts_on_success ?? 0).toFixed(2)}
                              </p>
                            </div>
                            <div className="bg-white border border-slate-200 rounded-lg p-3">
                              <p className="text-xs text-slate-500">Top-1</p>
                              <p className="font-semibold text-slate-900">{state.detail.summary.top1_success ?? benchmark.top1_success ?? 0}</p>
                            </div>
                            <div className="bg-white border border-slate-200 rounded-lg p-3">
                              <p className="text-xs text-slate-500">Top-3 / Top-5</p>
                              <p className="font-semibold text-slate-900">
                                {(state.detail.summary.top3_success ?? benchmark.top3_success ?? 0)} / {(state.detail.summary.top5_success ?? benchmark.top5_success ?? 0)}
                              </p>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Trace archives</p>
                            {archives.map((archive) => {
                              const isOpen = !!state.expandedArchives[archive.archive_id];
                              const limit = state.archiveRunLimit[archive.archive_id] ?? 20;
                              const runs = sortTraceRuns(archive.runs);
                              const visibleRuns = runs.slice(0, limit);

                              return (
                                <div key={archive.archive_id} className="bg-white border border-slate-200 rounded-lg">
                                  <button
                                    onClick={() => toggleArchive(benchmark.benchmark_id, archive.archive_id)}
                                    className="w-full px-4 py-3 flex items-center justify-between gap-4 text-left hover:bg-slate-50"
                                  >
                                    <div className="min-w-0">
                                      <p className="text-sm font-medium text-slate-900 truncate">{archive.archive_id}</p>
                                      <p className="text-xs text-slate-500">
                                        {archive.run_count} run(s) · {formatDateTime(archive.timestamp)} · success {formatPct(archive.success_rate)}
                                      </p>
                                    </div>
                                    <span className="text-xs text-slate-500">{isOpen ? 'Hide' : 'Show'}</span>
                                  </button>

                                  {isOpen && (
                                    <div className="border-t border-slate-200 p-4 space-y-3">
                                      <div className="grid grid-cols-3 gap-3 text-xs text-slate-600">
                                        <div>Verified rate: {formatPct(archive.verified_rate)}</div>
                                        <div>Avg attempts: {archive.avg_attempts_on_success.toFixed(2)}</div>
                                        <div>Runs: {archive.run_count}</div>
                                      </div>

                                      <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                          <thead>
                                            <tr className="border-b border-slate-200 text-xs text-slate-500">
                                              <th className="text-left py-2 pr-3">Run</th>
                                              <th className="text-left py-2 pr-3">Scenario</th>
                                              <th className="text-left py-2 pr-3">Timestamp</th>
                                              <th className="text-left py-2 pr-3">Attempts</th>
                                              <th className="text-left py-2 pr-3">Success</th>
                                              <th className="text-left py-2 pr-3"></th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {visibleRuns.map((run) => (
                                              <tr key={run.run_id} className="border-b border-slate-100">
                                                <td className="py-2 pr-3 font-mono text-xs">{run.run_id}</td>
                                                <td className="py-2 pr-3 text-xs text-slate-600">{run.scenario_id || 'n/a'}</td>
                                                <td className="py-2 pr-3 text-xs text-slate-600">{formatDateTime(run.timestamp)}</td>
                                                <td className="py-2 pr-3 text-xs text-slate-600">{run.total_attempts}</td>
                                                <td className={`py-2 pr-3 text-xs font-semibold ${run.success ? 'text-green-600' : 'text-red-600'}`}>
                                                  {run.success ? 'PASS' : 'FAIL'}
                                                </td>
                                                <td className="py-2 pr-3">
                                                  <button
                                                    onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                                                    className="text-xs text-blue-600 hover:text-blue-700"
                                                  >
                                                    Open
                                                  </button>
                                                </td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                      </div>

                                      {runs.length > visibleRuns.length && (
                                        <button
                                          onClick={() => loadMoreArchiveRuns(benchmark.benchmark_id, archive.archive_id)}
                                          className="text-xs font-medium text-slate-700 hover:text-slate-900"
                                        >
                                          Load more runs
                                        </button>
                                      )}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Direct results</h2>
              <p className="text-xs text-slate-500">
                Top-level `results/run_*.json` files only. Loaded in batches so the page stays responsive.
              </p>
            </div>
            <input
              value={runQuery}
              onChange={(e) => {
                setRunQuery(e.target.value);
                setDirectVisible(DIRECT_PAGE_SIZE);
              }}
              placeholder="Filter direct runs"
              className="w-full sm:w-72 px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-slate-400"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs text-slate-500">
                  <th className="text-left py-2 pr-3">Run</th>
                  <th className="text-left py-2 pr-3">Scenario</th>
                  <th className="text-left py-2 pr-3">Timestamp</th>
                  <th className="text-left py-2 pr-3">Attempts</th>
                  <th className="text-left py-2 pr-3">Generator</th>
                  <th className="text-left py-2 pr-3">Victim</th>
                  <th className="text-left py-2 pr-3">Result</th>
                  <th className="text-left py-2 pr-3"></th>
                </tr>
              </thead>
              <tbody>
                {visibleDirectRuns.map((run) => (
                  <tr key={run.run_id} className="border-b border-slate-100">
                    <td className="py-3 pr-3 font-mono text-xs">{run.run_id}</td>
                    <td className="py-3 pr-3 text-xs text-slate-600">{run.scenario_id || 'n/a'}</td>
                    <td className="py-3 pr-3 text-xs text-slate-600">{formatDateTime(run.timestamp)}</td>
                    <td className="py-3 pr-3 text-xs text-slate-600">{run.total_attempts}</td>
                    <td className="py-3 pr-3 text-xs text-slate-600 truncate max-w-[220px]">{run.generator || 'n/a'}</td>
                    <td className="py-3 pr-3 text-xs text-slate-600 truncate max-w-[220px]">{run.victim || 'n/a'}</td>
                    <td className={`py-3 pr-3 text-xs font-semibold ${run.success ? 'text-green-600' : 'text-red-600'}`}>
                      {run.success ? 'PASS' : 'FAIL'}
                    </td>
                    <td className="py-3 pr-3">
                      <button
                        onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                        className="text-xs text-blue-600 hover:text-blue-700"
                      >
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {hasMoreDirectRuns && (
            <div className="pt-4">
              <button
                onClick={loadMoreDirectRuns}
                className="px-4 py-2 text-sm font-medium bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200"
              >
                Load more direct runs
              </button>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
