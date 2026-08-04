import { memo, useEffect, useMemo, useState } from 'react';
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
  return date
    ? date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : 'n/a';
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

function StatusBadge({ success }: { success: boolean }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
        success
          ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
          : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'
      }`}
    >
      {success ? 'PASS' : 'FAIL'}
    </span>
  );
}

function Metric({ label, value, subtext }: { label: string; value: string; subtext?: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-stone-50 p-2.5 dark:border-stone-800 dark:bg-stone-900/50">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className="mt-0.5 font-display text-base font-semibold text-stone-900 dark:text-stone-100">{value}</p>
      {subtext && <p className="text-[10px] text-stone-500 dark:text-stone-500">{subtext}</p>}
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-stone-300 bg-stone-50 py-12 dark:border-stone-800 dark:bg-stone-900/30">
      <p className="text-sm text-stone-500 dark:text-stone-400">{message}</p>
    </div>
  );
}

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
            (benchmarksJson as BenchmarkListItem[]).map((item) => [
              item.benchmark_id,
              {
                expanded: false,
                loading: false,
                error: null,
                detail: null,
                expandedArchives: {},
                archiveRunLimit: {},
              },
            ]),
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
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center text-stone-500 dark:text-stone-400">
        <div className="flex items-center gap-3">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-300 border-t-teal-600 dark:border-stone-700 dark:border-t-teal-500" />
          Loading run explorer…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <main className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-7xl items-center justify-center px-6">
        <div className="w-full max-w-xl rounded-xl border border-rose-200 bg-white p-6 shadow-sm dark:border-rose-900/50 dark:bg-stone-900">
          <h1 className="font-display text-lg font-semibold text-rose-700 dark:text-rose-400">Run explorer unavailable</h1>
          <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-stone-800 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            Retry
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[1600px] p-4 lg:p-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Benchmark folders */}
        <section className="lg:col-span-2 space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h1 className="font-display text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
                Benchmark folders
              </h1>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                Expand a benchmark to view trace archives and per-run rows.
              </p>
            </div>
            <input
              value={benchmarkQuery}
              onChange={(e) => setBenchmarkQuery(e.target.value)}
              placeholder="Filter benchmarks"
              className="input w-full sm:w-72"
            />
          </div>

          {filteredBenchmarks.length === 0 ? (
            <EmptyState message="No benchmarks match your filters." />
          ) : (
            <div className="space-y-3">
              {filteredBenchmarks.map((benchmark) => {
                const state = benchmarkStates[benchmark.benchmark_id];
                if (!state) return null;
                const summary = state.detail?.summary ?? benchmark;
                const archives = sortTraceArchives(state.detail?.trace_archives ?? []);

                return (
                  <div
                    key={benchmark.benchmark_id}
                    className="overflow-hidden rounded-xl border border-stone-200 bg-white shadow-sm transition-shadow hover:shadow-md dark:border-stone-800 dark:bg-stone-900"
                  >
                    <button
                      onClick={() => expandBenchmark(benchmark.benchmark_id)}
                      className="flex w-full items-center justify-between gap-4 px-4 py-3.5 text-left transition-colors hover:bg-stone-50 dark:hover:bg-stone-800/50"
                    >
                      <div className="min-w-0">
                        <p className="font-mono text-xs text-teal-700 dark:text-teal-400 truncate">{benchmark.benchmark_id}</p>
                        <p className="mt-0.5 text-sm text-stone-600 dark:text-stone-300">
                          {formatDateTime(benchmark.timestamp)} · {benchmark.trace_archive_count} archive(s)
                        </p>
                      </div>
                      <div className="flex items-center gap-4 text-right">
                        <div>
                          <div className="font-display text-sm font-semibold text-stone-900 dark:text-stone-100">
                            {formatPct(summary.success_rate)}
                          </div>
                          <div className="text-xs text-stone-500 dark:text-stone-500">
                            {summary.total_rounds ?? benchmark.total_rounds ?? 0} rounds
                          </div>
                        </div>
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className={`h-4 w-4 text-stone-400 transition-transform ${state.expanded ? 'rotate-180' : ''}`}
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        >
                          <polyline points="6 9 12 15 18 9" />
                        </svg>
                      </div>
                    </button>

                    {state.expanded && (
                      <div className="border-t border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-950/50">
                        {state.loading && <p className="text-sm text-stone-500 dark:text-stone-400">Loading benchmark detail…</p>}
                        {state.error && <p className="text-sm text-rose-600 dark:text-rose-400">{state.error}</p>}
                        {state.detail && (
                          <div className="space-y-4">
                            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                              <Metric label="Success rate" value={formatPct(state.detail.summary.success_rate ?? benchmark.success_rate)} />
                              <Metric
                                label="Verified success"
                                value={String(state.detail.summary.verified_success ?? benchmark.verified_success ?? 0)}
                              />
                              <Metric
                                label="Avg attempts"
                                value={Number(state.detail.summary.avg_attempts_on_success ?? benchmark.avg_attempts_on_success ?? 0).toFixed(2)}
                              />
                              <Metric label="Top-1" value={String(state.detail.summary.top1_success ?? benchmark.top1_success ?? 0)} />
                              <Metric
                                label="Top-3 / Top-5"
                                value={`${state.detail.summary.top3_success ?? benchmark.top3_success ?? 0} / ${state.detail.summary.top5_success ?? benchmark.top5_success ?? 0}`}
                              />
                            </div>

                            <div className="space-y-2">
                              <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">
                                Trace archives
                              </p>
                              {archives.map((archive) => {
                                const isOpen = !!state.expandedArchives[archive.archive_id];
                                const limit = state.archiveRunLimit[archive.archive_id] ?? 20;
                                const runs = sortTraceRuns(archive.runs);
                                const visibleRuns = runs.slice(0, limit);

                                return (
                                  <div
                                    key={archive.archive_id}
                                    className="overflow-hidden rounded-lg border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900"
                                  >
                                    <button
                                      onClick={() => toggleArchive(benchmark.benchmark_id, archive.archive_id)}
                                      className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition-colors hover:bg-stone-50 dark:hover:bg-stone-800/50"
                                    >
                                      <div className="min-w-0">
                                        <p className="text-sm font-semibold text-stone-900 dark:text-stone-100 truncate">{archive.archive_id}</p>
                                        <p className="text-xs text-stone-500 dark:text-stone-400">
                                          {archive.run_count} runs · {formatDateTime(archive.timestamp)} · success {formatPct(archive.success_rate)}
                                        </p>
                                      </div>
                                      <span className="text-xs font-medium text-stone-500 dark:text-stone-400">
                                        {isOpen ? 'Hide' : 'Show'}
                                      </span>
                                    </button>

                                    {isOpen && (
                                      <div className="border-t border-stone-200 p-4 dark:border-stone-800">
                                        <div className="mb-3 grid grid-cols-3 gap-3 text-xs text-stone-600 dark:text-stone-400">
                                          <div>Verified rate: <span className="font-semibold text-stone-900 dark:text-stone-100">{formatPct(archive.verified_rate)}</span></div>
                                          <div>Avg attempts: <span className="font-semibold text-stone-900 dark:text-stone-100">{archive.avg_attempts_on_success.toFixed(2)}</span></div>
                                          <div>Runs: <span className="font-semibold text-stone-900 dark:text-stone-100">{archive.run_count}</span></div>
                                        </div>

                                        <div className="overflow-x-auto rounded-lg border border-stone-200 dark:border-stone-800">
                                          <table className="w-full text-sm">
                                            <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
                                              <tr>
                                                <th className="px-3 py-2">Run</th>
                                                <th className="px-3 py-2">Scenario</th>
                                                <th className="px-3 py-2">Attempts</th>
                                                <th className="px-3 py-2">Result</th>
                                                <th className="px-3 py-2"></th>
                                              </tr>
                                            </thead>
                                            <tbody className="divide-y divide-stone-100 dark:divide-stone-800">
                                              {visibleRuns.map((run) => (
                                                <tr
                                                  key={run.run_id}
                                                  className="transition-colors hover:bg-stone-50 dark:hover:bg-stone-800/50"
                                                >
                                                  <td className="px-3 py-2 font-mono text-xs text-stone-700 dark:text-stone-300">{run.run_id}</td>
                                                  <td className="px-3 py-2 text-xs text-stone-600 dark:text-stone-400">{run.scenario_id || 'n/a'}</td>
                                                  <td className="px-3 py-2 text-xs text-stone-600 dark:text-stone-400">{run.total_attempts}</td>
                                                  <td className="px-3 py-2"><StatusBadge success={run.success} /></td>
                                                  <td className="px-3 py-2">
                                                    <button
                                                      onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                                                      className="text-xs font-medium text-teal-700 hover:text-teal-800 dark:text-teal-400 dark:hover:text-teal-300"
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
                                            className="mt-3 text-xs font-medium text-stone-700 hover:text-stone-900 dark:text-stone-400 dark:hover:text-stone-200"
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
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Direct results */}
        <section className="space-y-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="font-display text-xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">Direct results</h2>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                Top-level result files. Loaded in batches.
              </p>
            </div>
            <input
              value={runQuery}
              onChange={(e) => {
                setRunQuery(e.target.value);
                setDirectVisible(DIRECT_PAGE_SIZE);
              }}
              placeholder="Filter direct runs"
              className="input w-full"
            />
          </div>

          <div className="overflow-hidden rounded-xl border border-stone-200 bg-white shadow-sm dark:border-stone-800 dark:bg-stone-900">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
                  <tr>
                    <th className="px-4 py-3">Run</th>
                    <th className="px-4 py-3">Scenario</th>
                    <th className="px-4 py-3">Victim</th>
                    <th className="px-4 py-3">Result</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100 dark:divide-stone-800">
                  {visibleDirectRuns.map((run) => (
                    <tr
                      key={run.run_id}
                      className="transition-colors hover:bg-stone-50 dark:hover:bg-stone-800/50"
                    >
                      <td className="px-4 py-3">
                        <div className="font-mono text-xs text-stone-700 dark:text-stone-300">{run.run_id}</div>
                        <div className="text-[10px] text-stone-500 dark:text-stone-500">{formatDateTime(run.timestamp)}</div>
                      </td>
                      <td className="px-4 py-3 text-xs text-stone-600 dark:text-stone-400">{run.scenario_id || 'n/a'}</td>
                      <td className="px-4 py-3 text-xs text-stone-600 dark:text-stone-400 truncate max-w-[160px]" title={run.victim}>
                        {(run.victim || 'n/a').split('/').pop()}
                      </td>
                      <td className="px-4 py-3"><StatusBadge success={run.success} /></td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                          className="rounded-md px-2 py-1 text-xs font-medium text-teal-700 transition-colors hover:bg-teal-50 dark:text-teal-400 dark:hover:bg-teal-950/30"
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
              <div className="border-t border-stone-200 p-3 dark:border-stone-800">
                <button
                  onClick={loadMoreDirectRuns}
                  className="btn-default w-full"
                >
                  Load more direct runs ({filteredDirectRuns.length - visibleDirectRuns.length} remaining)
                </button>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
