import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  PieChart,
  Pie,
} from 'recharts';
import type {
  BenchmarkDetail,
  BenchmarkListItem,
  TraceRunListItem,
  MutationFallbackDiagnostics,
  PerMutatorDiagnostics,
  SuccessPathBreakdownEntry,
  FailureModeBreakdownEntry,
} from '../types/autored';

const numberFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const pctFmt = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

const asDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatPct = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return '0.0%';
  return `${pctFmt.format(Number(value) * 100)}%`;
};

const formatDelta = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return 'baseline n/a';
  const delta = Number(value);
  const prefix = delta > 0 ? '+' : '';
  return `${prefix}${pctFmt.format(delta * 100)} pp`;
};

const formatNumber = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return 'n/a';
  return numberFmt.format(Number(value));
};

const getArchiveModel = (archiveId: string) => {
  const segment = archiveId.split('/')[1];
  return segment ? segment.replace(/--/g, '/') : archiveId;
};

const sortBenchmarks = (items: BenchmarkListItem[]) =>
  [...items].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return b.benchmark_id.localeCompare(a.benchmark_id);
  });

const sortTraceRuns = (runs: TraceRunListItem[]) =>
  [...runs].sort((a, b) => {
    const ta = asDate(a.timestamp)?.getTime() ?? 0;
    const tb = asDate(b.timestamp)?.getTime() ?? 0;
    if (ta !== tb) return tb - ta;
    return a.run_id.localeCompare(b.run_id);
  });

interface Metrics {
  total: number;
  successRate: number;
  verifiedCount: number;
  verifiedRate: number;
  avgAttempts: number;
  top1: number;
  top3: number;
  top5: number;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  isFiltered: boolean;
}

export default function BenchmarkDashboard() {
  const navigate = useNavigate();
  const { benchmarkId } = useParams<{ benchmarkId?: string }>();
  const [benchmarks, setBenchmarks] = useState<BenchmarkListItem[]>([]);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string>('');
  const [benchmarkDetail, setBenchmarkDetail] = useState<BenchmarkDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runFilter, setRunFilter] = useState('');
  const [archiveFilter, setArchiveFilter] = useState<'all' | string>('all');
  const [modelFilter, setModelFilter] = useState<'all' | string>('all');

  useEffect(() => {
    let cancelled = false;
    const loadBenchmarks = async () => {
      try {
        const res = await fetch('/api/benchmarks');
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = sortBenchmarks((await res.json()) as BenchmarkListItem[]);
        if (cancelled) return;
        setBenchmarks(data);
        setError(null);
        const fallbackId = data[0]?.benchmark_id ?? '';
        const nextId = benchmarkId && data.some((item) => item.benchmark_id === benchmarkId)
          ? benchmarkId
          : fallbackId;
        setSelectedBenchmarkId(nextId);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load benchmarks');
      } finally {
        if (!cancelled) setLoadingList(false);
      }
    };
    loadBenchmarks();
    return () => { cancelled = true; };
  }, [benchmarkId]);

  useEffect(() => {
    if (!selectedBenchmarkId) return;
    if (!benchmarkId || benchmarkId !== selectedBenchmarkId) {
      navigate(`/benchmarks/${encodeURIComponent(selectedBenchmarkId)}`, { replace: true });
    }
  }, [benchmarkId, navigate, selectedBenchmarkId]);

  useEffect(() => {
    if (!selectedBenchmarkId) return;
    let cancelled = false;
    const loadDetail = async () => {
      setLoadingDetail(true);
      try {
        const res = await fetch(`/api/benchmarks/${encodeURIComponent(selectedBenchmarkId)}`);
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = (await res.json()) as BenchmarkDetail;
        if (cancelled) return;
        setBenchmarkDetail(data);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setBenchmarkDetail(null);
        setError(err instanceof Error ? err.message : 'Failed to load benchmark detail');
      } finally {
        if (!cancelled) setLoadingDetail(false);
      }
    };
    loadDetail();
    return () => { cancelled = true; };
  }, [selectedBenchmarkId]);

  const currentBenchmark = useMemo(
    () => benchmarks.find((item) => item.benchmark_id === selectedBenchmarkId) ?? null,
    [benchmarks, selectedBenchmarkId],
  );

  const selectedIndex = useMemo(
    () => benchmarks.findIndex((item) => item.benchmark_id === selectedBenchmarkId),
    [benchmarks, selectedBenchmarkId],
  );

  const previousBenchmark = selectedIndex >= 0 ? benchmarks[selectedIndex + 1] ?? null : null;

  const summary = benchmarkDetail?.summary ?? currentBenchmark ?? null;
  const traceArchives = benchmarkDetail?.trace_archives ?? [];
  const traceRuns = useMemo(() => sortTraceRuns(benchmarkDetail?.trace_runs ?? []), [benchmarkDetail]);

  const activeFilter = modelFilter !== 'all' || archiveFilter !== 'all';

  const modelOptions = useMemo(() => {
    const victims = traceRuns.map((run) => run.victim).filter(Boolean);
    return [...new Set(victims)].sort();
  }, [traceRuns]);

  const filteredArchives = useMemo(() => {
    if (modelFilter === 'all') return traceArchives;
    return traceArchives.filter((archive) => getArchiveModel(archive.archive_id) === modelFilter);
  }, [modelFilter, traceArchives]);

  const filteredTraceRuns = useMemo(() => {
    const query = runFilter.trim().toLowerCase();
    return traceRuns.filter((run) => {
      if (modelFilter !== 'all' && run.victim !== modelFilter) return false;
      if (archiveFilter !== 'all' && !run.file_path.includes(archiveFilter)) return false;
      if (!query) return true;
      return [
        run.run_id,
        run.scenario_id,
        run.access_code,
        run.generator,
        run.victim,
        String(run.worker_id ?? ''),
        run.timestamp,
      ].some((field) => field.toLowerCase().includes(query));
    });
  }, [archiveFilter, modelFilter, runFilter, traceRuns]);

  const derivedMetrics: Metrics = useMemo(() => {
    const runs = filteredTraceRuns;
    const total = runs.length;
    if (!total) {
      return { total: 0, successRate: 0, verifiedCount: 0, verifiedRate: 0, avgAttempts: 0, top1: 0, top3: 0, top5: 0, f1: null, precision: null, recall: null, isFiltered: true };
    }
    const successes = runs.filter((r) => r.success).length;
    const verifiedCount = runs.filter((r) => r.verified_success).length;
    const successfulRuns = runs.filter((r) => r.success);
    const avgAttempts = successfulRuns.length
      ? successfulRuns.reduce((sum, r) => sum + r.total_attempts, 0) / successfulRuns.length
      : 0;
    const top = (n: number) => successfulRuns.filter((r) => r.total_attempts <= n).length;
    return {
      total,
      successRate: total ? successes / total : 0,
      verifiedCount,
      verifiedRate: total ? verifiedCount / total : 0,
      avgAttempts,
      top1: top(1),
      top3: top(3),
      top5: top(5),
      f1: null,
      precision: null,
      recall: null,
      isFiltered: true,
    };
  }, [filteredTraceRuns]);

  const globalMetrics: Metrics = useMemo(() => {
    const total = summary?.total_rounds ?? 0;
    const verifiedCount = summary?.verified_success ?? 0;
    return {
      total,
      successRate: summary ? Number(summary.success_rate ?? 0) : 0,
      verifiedCount,
      verifiedRate: total ? verifiedCount / total : 0,
      avgAttempts: summary ? Number(summary.avg_attempts_on_success ?? 0) : 0,
      top1: summary ? Number(summary.top1_success ?? 0) : 0,
      top3: summary ? Number(summary.top3_success ?? 0) : 0,
      top5: summary ? Number(summary.top5_success ?? 0) : 0,
      f1: summary?.extractor_metrics?.f1 ?? null,
      precision: summary?.extractor_metrics?.precision ?? null,
      recall: summary?.extractor_metrics?.recall ?? null,
      isFiltered: false,
    };
  }, [summary]);

  const metrics = activeFilter ? derivedMetrics : globalMetrics;

  const previousSummary = activeFilter
    ? null
    : previousBenchmark
    ? {
        success_rate: previousBenchmark.success_rate,
        verified_rate: previousBenchmark.total_rounds
          ? previousBenchmark.verified_success / previousBenchmark.total_rounds
          : 0,
        avg_attempts: previousBenchmark.avg_attempts_on_success,
        top1: previousBenchmark.top1_success,
        top3: previousBenchmark.top3_success,
        top5: previousBenchmark.top5_success,
        f1: previousBenchmark.extractor_metrics?.f1 ?? 0,
      }
    : null;

  const currentVsPrevious = currentBenchmark && previousSummary && !activeFilter
    ? {
        success_rate: globalMetrics.successRate - previousSummary.success_rate,
        verified_rate: globalMetrics.verifiedRate - previousSummary.verified_rate,
        avg_attempts: globalMetrics.avgAttempts - previousSummary.avg_attempts,
        top1: globalMetrics.top1 - previousSummary.top1,
        top3: globalMetrics.top3 - previousSummary.top3,
        top5: globalMetrics.top5 - previousSummary.top5,
        f1: Number(globalMetrics.f1 ?? 0) - previousSummary.f1,
      }
    : null;

  const workerChartData = useMemo(() => {
    return (benchmarkDetail?.worker_summaries ?? []).map((worker) => ({
      worker: `Worker ${worker.worker_id}`,
      successRate: Number(worker.success_rate ?? 0) * 100,
      successes: worker.successes,
      rounds: worker.rounds,
    }));
  }, [benchmarkDetail]);

  const archivePieData = useMemo(() => {
    return filteredArchives.map((archive) => ({ name: archive.archive_id, value: archive.run_count }));
  }, [filteredArchives]);

  if (loadingList) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center text-stone-500 dark:text-stone-400">
        <div className="flex items-center gap-3">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-300 border-t-teal-600 dark:border-stone-700 dark:border-t-teal-500" />
          Loading benchmark explorer…
        </div>
      </div>
    );
  }

  if (!benchmarks.length) {
    return (
      <main className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-7xl items-center justify-center p-6">
        <div className="text-center">
          <p className="font-display text-lg text-stone-900 dark:text-stone-100">No benchmark summaries found</p>
          <p className="mt-2 text-sm text-stone-500 dark:text-stone-400">Run a benchmark to populate results.</p>
          <button
            onClick={() => navigate('/runs')}
            className="mt-4 text-sm font-medium text-teal-700 hover:text-teal-800 dark:text-teal-400 dark:hover:text-teal-300"
          >
            Back to Runs
          </button>
        </div>
      </main>
    );
  }

  const modelDisplayName = modelFilter === 'all' ? 'All models' : modelFilter.split('/').pop() ?? modelFilter;

  return (
    <main className="mx-auto max-w-[1600px] p-4 lg:p-6">
      {error && (
        <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/20 dark:text-rose-400">
          {error}
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">Benchmark Explorer</h1>
          <p className="text-sm text-stone-500 dark:text-stone-400">Summary folders, worker rollups, trace archives, and per-run drilldown.</p>
        </div>
        <select
          value={selectedBenchmarkId}
          onChange={(e) => setSelectedBenchmarkId(e.target.value)}
          className="input sm:w-96"
        >
          {sortBenchmarks(benchmarks).map((item) => (
            <option key={item.benchmark_id} value={item.benchmark_id}>{item.benchmark_id}</option>
          ))}
        </select>
      </div>

      <section className="mb-4 flex flex-wrap items-center gap-3">
        <span className="text-sm text-stone-500 dark:text-stone-400">Scope</span>
        <select
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          className="input"
        >
          <option value="all">All models</option>
          {modelOptions.map((model) => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
        <select
          value={archiveFilter}
          onChange={(e) => setArchiveFilter(e.target.value)}
          className="input"
        >
          <option value="all">All archives</option>
          {filteredArchives.map((archive) => (
            <option key={archive.archive_id} value={archive.archive_id}>{archive.archive_id}</option>
          ))}
        </select>
        {activeFilter && (
          <button
            onClick={() => { setModelFilter('all'); setArchiveFilter('all'); }}
            className="btn-default text-xs"
          >
            Clear filters
          </button>
        )}
        <span className="ml-auto text-xs text-stone-500 dark:text-stone-400">
          {activeFilter ? `Metrics reflect ${modelDisplayName}${archiveFilter !== 'all' ? ` · ${archiveFilter.split('/').pop()}` : ''}` : 'Metrics reflect full benchmark'}
        </span>
      </section>

      <section className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Success Rate"
          value={formatPct(metrics.successRate)}
          subtext={`${formatNumber(activeFilter ? filteredTraceRuns.filter((r) => r.success).length : summary?.total_successes ?? 0)}/${formatNumber(metrics.total)} rounds`}
          delta={currentVsPrevious ? formatDelta(currentVsPrevious.success_rate) : activeFilter ? 'filtered scope' : 'baseline n/a'}
          positive={currentVsPrevious ? currentVsPrevious.success_rate >= 0 : true}
        />
        <MetricCard
          label="Verified Success"
          value={formatNumber(metrics.verifiedCount)}
          subtext={formatPct(metrics.verifiedRate)}
          delta={currentVsPrevious ? formatDelta(currentVsPrevious.verified_rate) : activeFilter ? 'filtered scope' : 'baseline n/a'}
          positive={currentVsPrevious ? currentVsPrevious.verified_rate >= 0 : true}
        />
        <MetricCard
          label="Avg Attempts"
          value={formatNumber(metrics.avgAttempts)}
          subtext="on successful rounds"
          delta={currentVsPrevious ? `${currentVsPrevious.avg_attempts > 0 ? '+' : ''}${numberFmt.format(currentVsPrevious.avg_attempts)}` : activeFilter ? 'filtered scope' : 'baseline n/a'}
          positive={currentVsPrevious ? currentVsPrevious.avg_attempts <= 0 : true}
        />
        <MetricCard
          label="Top-1 / Top-3 / Top-5"
          value={`${formatNumber(metrics.top1)} / ${formatNumber(metrics.top3)} / ${formatNumber(metrics.top5)}`}
          subtext="rounds recovered by rank"
          delta={currentVsPrevious ? `Δ F1 ${formatDelta(currentVsPrevious.f1)}` : metrics.isFiltered ? 'filtered scope' : 'baseline n/a'}
          positive={currentVsPrevious ? currentVsPrevious.f1 >= 0 : true}
        />
        <MetricCard
          label="Extractor F1"
          value={metrics.f1 !== null ? formatPct(metrics.f1) : '—'}
          subtext={metrics.f1 !== null ? `P ${formatPct(metrics.precision)} · R ${formatPct(metrics.recall)}` : 'global benchmark only'}
          delta={currentVsPrevious ? formatDelta(currentVsPrevious.f1) : activeFilter ? 'filtered scope' : 'baseline n/a'}
          positive={currentVsPrevious ? currentVsPrevious.f1 >= 0 : true}
        />
        <MetricCard
          label="Trace Runs"
          value={formatNumber(filteredTraceRuns.length)}
          subtext={`${filteredArchives.length} archive${filteredArchives.length === 1 ? '' : 's'}${modelFilter !== 'all' ? ` · ${modelDisplayName}` : ''}`}
          delta={loadingDetail ? 'refreshing…' : 'ready'}
        />
      </section>

      <section className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Panel title="Benchmark Metadata" className="xl:col-span-1">
          <DetailRow label="Benchmark ID" value={selectedBenchmarkId || 'unknown'} mono />
          <DetailRow label="Timestamp" value={summary?.metadata?.timestamp ?? currentBenchmark?.timestamp ?? 'unknown'} />
          <DetailRow label="Target Model" value={summary?.metadata?.target_model ?? 'unknown'} />
          <DetailRow label="Rounds" value={formatNumber(summary?.metadata?.n_rounds ?? summary?.total_rounds ?? 0)} />
          <DetailRow label="Workers" value={formatNumber(summary?.metadata?.num_workers ?? summary?.worker_summaries?.length ?? 0)} />
          <DetailRow label="Max Interactions" value={formatNumber(summary?.metadata?.max_interactions ?? 'n/a')} />
          <DetailRow label="Merged From" value={Array.isArray(summary?.metadata?.merged_from) ? summary.metadata.merged_from.join(', ') : 'n/a'} />
          <DetailRow label="Trace Archives" value={traceArchives.map((archive) => archive.archive_id).join(', ') || 'n/a'} />
        </Panel>

        <Panel title="Worker Breakdown" className="xl:col-span-2">
          <div className="h-64">
            {workerChartData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={workerChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-stone-200 dark:text-stone-800" />
                  <XAxis dataKey="worker" tick={{ fontSize: 12, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                  <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'var(--elevated)', borderColor: 'var(--border)', borderRadius: '0.5rem' }}
                    itemStyle={{ color: 'currentColor' }}
                  />
                  <Legend />
                  <Bar dataKey="successRate" fill="#0d9488" name="Success Rate %" />
                  <Bar dataKey="rounds" fill="#a8a29e" name="Rounds" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-stone-500 dark:text-stone-400">Worker summaries are not available for this benchmark.</p>
            )}
          </div>
          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
            {(benchmarkDetail?.worker_summaries ?? []).map((worker) => (
              <div key={worker.worker_id} className="rounded-lg border border-stone-200 p-3 dark:border-stone-800">
                <div className="flex items-center justify-between">
                  <p className="font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Worker {worker.worker_id}</p>
                  <span className="font-mono text-sm text-stone-600 dark:text-stone-400">{formatPct(worker.success_rate)}</span>
                </div>
                <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">{worker.successes}/{worker.rounds} successful rounds</p>
              </div>
            ))}
          </div>
        </Panel>
      </section>

      {/* JailGuard Mutation Fallback panel — only renders when fallback ran */}
      {summary && <FallbackPanel summary={summary} />}

      <section className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Panel title="Trace Archives" className="xl:col-span-1">
          <div className="space-y-2">
            {filteredArchives.length ? filteredArchives.map((archive) => (
              <button
                key={archive.archive_id}
                onClick={() => setArchiveFilter(archiveFilter === archive.archive_id ? 'all' : archive.archive_id)}
                className={`w-full rounded-lg border p-3 text-left transition-colors ${
                  archiveFilter === archive.archive_id
                    ? 'border-teal-400 bg-teal-50 dark:border-teal-600 dark:bg-teal-950/30'
                    : 'border-stone-200 hover:bg-stone-50 dark:border-stone-800 dark:hover:bg-stone-800/50'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-semibold text-stone-900 dark:text-stone-100">{archive.archive_id}</p>
                  <span className="text-xs text-stone-500 dark:text-stone-400">{archive.run_count} runs</span>
                </div>
                <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">
                  Success {formatPct(archive.success_rate)} · Verified {formatPct(archive.verified_rate)} · Avg {formatNumber(archive.avg_attempts_on_success)}
                </p>
              </button>
            )) : (
              <p className="text-sm text-stone-500 dark:text-stone-400">No trace archives match the current model filter.</p>
            )}
          </div>
        </Panel>

        <Panel title="Archive Composition" className="xl:col-span-2">
          <div className="h-64">
            {archivePieData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={archivePieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={3}
                  >
                    {archivePieData.map((entry, index) => (
                      <Cell key={entry.name} fill={['#0d9488', '#14b8a6', '#2dd4bf', '#5eead4', '#99f6e4'][index % 5]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: 'var(--elevated)', borderColor: 'var(--border)', borderRadius: '0.5rem' }} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-stone-500 dark:text-stone-400">No archive distribution data available.</p>
            )}
          </div>
        </Panel>
      </section>

      <section className="mt-6">
        <Panel title="Run-Level Drilldown">
          <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center">
            <input
              value={runFilter}
              onChange={(e) => setRunFilter(e.target.value)}
              placeholder="Filter by run id, scenario, worker, access code, generator, or victim"
              className="input lg:flex-1"
            />
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="input"
            >
              <option value="all">All models</option>
              {modelOptions.map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            <select
              value={archiveFilter}
              onChange={(e) => setArchiveFilter(e.target.value)}
              className="input"
            >
              <option value="all">All archives</option>
              {filteredArchives.map((archive) => (
                <option key={archive.archive_id} value={archive.archive_id}>{archive.archive_id}</option>
              ))}
            </select>
            <div className="text-sm text-stone-500 dark:text-stone-400 lg:ml-auto">
              {filteredTraceRuns.length}/{traceRuns.length} runs shown
            </div>
          </div>

          <div className="overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
                <tr>
                  <Th>Run</Th>
                  <Th>Scenario</Th>
                  <Th>Archive</Th>
                  <Th>Worker</Th>
                  <Th>Attempts</Th>
                  <Th>Status</Th>
                  <Th>Access Code</Th>
                  <Th>Generator</Th>
                  <Th>Victim</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 bg-white dark:divide-stone-800 dark:bg-stone-900">
                {filteredTraceRuns.map((run) => (
                  <tr
                    key={`${run.run_id}-${run.file_path}`}
                    onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                    className="cursor-pointer transition-colors hover:bg-stone-50 dark:hover:bg-stone-800/50"
                  >
                    <Td mono>{run.run_id}</Td>
                    <Td mono>{run.scenario_id || 'unknown'}</Td>
                    <Td>{run.file_path.split('/').slice(-2, -1)[0] || 'unknown'}</Td>
                    <Td>{run.worker_id ?? 'n/a'}</Td>
                    <Td>{run.attempt_count ?? run.total_attempts}</Td>
                    <Td>
                      <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${
                        run.success
                          ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
                          : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'
                      }`}>
                        {run.verified_success ? 'verified' : run.success ? 'success' : 'failed'}
                      </span>
                    </Td>
                    <Td className="font-mono text-amber-700 dark:text-amber-400">{run.access_code || 'n/a'}</Td>
                    <Td>{run.generator || 'n/a'}</Td>
                    <Td>{run.victim || 'n/a'}</Td>
                  </tr>
                ))}
                {!filteredTraceRuns.length && (
                  <tr>
                    <td colSpan={9} className="px-4 py-8 text-center text-sm text-stone-500 dark:text-stone-400">
                      No trace runs match the current filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </section>

      <section className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Current Summary Payload">
          <pre className="max-h-96 overflow-auto rounded-lg bg-stone-950 p-4 text-xs text-stone-100">
            {JSON.stringify(summary, null, 2)}
          </pre>
        </Panel>
        <Panel title="Selected Benchmark Trace Payload">
          <pre className="max-h-96 overflow-auto rounded-lg bg-stone-950 p-4 text-xs text-stone-100">
            {JSON.stringify(benchmarkDetail, null, 2)}
          </pre>
        </Panel>
      </section>

      {loadingDetail && (
        <div className="mt-4 text-sm text-stone-500 dark:text-stone-400">Refreshing benchmark detail…</div>
      )}
    </main>
  );
}

function Panel(props: { title: string; className?: string; children: ReactNode }) {
  return (
    <section className={`rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900 ${props.className ?? ''}`}>
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="font-display text-sm font-semibold text-stone-900 dark:text-stone-100">{props.title}</h2>
      </div>
      {props.children}
    </section>
  );
}

function MetricCard(props: { label: string; value: string; subtext: string; delta: string; positive?: boolean }) {
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{props.label}</p>
      <p className="mt-1 break-words font-display text-xl font-semibold text-stone-900 dark:text-stone-100">{props.value}</p>
      <p className="mt-0.5 text-xs text-stone-500 dark:text-stone-400">{props.subtext}</p>
      <p className={`mt-2 text-xs font-medium ${
        props.delta.includes('baseline') || props.delta.includes('global') || props.delta.includes('filtered')
          ? 'text-stone-400 dark:text-stone-500'
          : props.positive !== false
          ? 'text-emerald-600 dark:text-emerald-400'
          : 'text-rose-600 dark:text-rose-400'
      }`}>
        {props.delta}
      </p>
    </div>
  );
}

function DetailRow(props: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="border-b border-stone-100 py-2 last:border-b-0 dark:border-stone-800">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{props.label}</p>
      <p className={`mt-0.5 text-sm text-stone-900 dark:text-stone-100 ${props.mono ? 'break-all font-mono' : 'break-words'}`}>
        {props.value}
      </p>
    </div>
  );
}

// --- JailGuard Mutation Fallback panel (Change 2) ---
// Renders the fallback diagnostics from merged_summary.json when mutation
// fallback ran. Conditionally shown: hidden for benchmarks that didn't use
// fallback (mutation_fallback_triggered falsy AND mutation_fallback_enabled
// not true), so non-fallback runs stay uncluttered.

const MUTATOR_ORDER = ['EN', 'PI', 'SR', 'TL'] as const;

function FallbackPanel({ summary }: { summary: Record<string, any> }) {
  const triggered: number = Number(summary?.mutation_fallback_triggered) || 0;
  const successes: number = Number(summary?.mutation_fallback_successes) || 0;
  const enabled: boolean = Boolean(summary?.metadata?.mutation_fallback_enabled);
  if (!triggered && !enabled) return null; // non-fallback benchmark -> hide

  const diag = summary?.mutation_fallback_diagnostics as MutationFallbackDiagnostics | undefined;
  const conversion = triggered > 0 ? (successes / triggered) * 100 : 0;
  const noOpRate = diag?.no_op_rate ?? 0;
  const mutatorCounts: Record<string, number> = diag?.mutator_counts ?? {};
  const perMutator: Record<string, PerMutatorDiagnostics> = diag?.per_mutator ?? {};
  const successPaths: SuccessPathBreakdownEntry[] = summary?.success_path_breakdown ?? [];
  const failureModes: FailureModeBreakdownEntry[] = summary?.failure_mode_breakdown ?? [];
  const meta = summary?.metadata ?? {};

  // Mutator draw-distribution bar chart data, ordered EN/PI/SR/TL then any extra.
  const drawData = Object.keys(mutatorCounts)
    .sort((a, b) => MUTATOR_ORDER.indexOf(a as any) - MUTATOR_ORDER.indexOf(b as any))
    .map((m) => ({ mutator: m, drawn: mutatorCounts[m] }));

  return (
    <Panel title="JailGuard Mutation Fallback" className="xl:col-span-3">
      {/* Summary metric row */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard label="Fallback Triggered" value={formatNumber(triggered)} subtext="scenarios that entered fallback" delta="fallback" />
        <MetricCard label="Fallback Successes" value={formatNumber(successes)} subtext="scenarios cracked by a variant" delta="fallback" />
        <MetricCard label="Conversion Rate" value={pctFmt.format(conversion)} subtext="successes ÷ triggered" delta={conversion >= 20 ? 'above 20%' : 'below 20%'} positive={conversion >= 20} />
        <MetricCard label="No-Op Rate" value={formatPct(noOpRate)} subtext="variants == seed (wasted queries)" delta={noOpRate <= 0.05 ? 'healthy' : 'check mutators'} positive={noOpRate <= 0.05} />
      </div>

      {/* Per-mutator table + draw distribution */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Per-Mutator Attribution</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-stone-200 text-left text-xs uppercase tracking-wider text-stone-500 dark:border-stone-800 dark:text-stone-400">
                  <th className="py-2 pr-3">Mutator</th>
                  <th className="py-2 pr-3">Drawn</th>
                  <th className="py-2 pr-3">Wins</th>
                  <th className="py-2 pr-3">Win Rate</th>
                  <th className="py-2 pr-3">No-Op Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(perMutator)
                  .sort((a, b) => MUTATOR_ORDER.indexOf(a as any) - MUTATOR_ORDER.indexOf(b as any))
                  .map((m) => {
                    const pm = perMutator[m];
                    return (
                      <tr key={m} className="border-b border-stone-100 dark:border-stone-800">
                        <td className="py-2 pr-3 font-mono text-stone-900 dark:text-stone-100">{m}</td>
                        <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{formatNumber(pm.drawn)}</td>
                        <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{formatNumber(pm.wins)}</td>
                        <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(pm.win_rate * 100)}%</td>
                        <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(pm.no_op_rate * 100)}%</td>
                      </tr>
                    );
                  })}
                {Object.keys(perMutator).length === 0 && (
                  <tr><td colSpan={5} className="py-3 text-stone-500 dark:text-stone-400">No per-mutator diagnostics available.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Mutator Draw Distribution</h3>
          <div className="h-48">
            {drawData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={drawData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-stone-200 dark:text-stone-800" />
                  <XAxis dataKey="mutator" tick={{ fontSize: 12, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                  <YAxis tick={{ fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--elevated)', borderColor: 'var(--border)', borderRadius: '0.5rem' }} itemStyle={{ color: 'currentColor' }} />
                  <Bar dataKey="drawn" fill="#0d9488" name="Variants drawn" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-stone-500 dark:text-stone-400">No mutator draw data.</p>
            )}
          </div>
        </div>
      </div>

      {/* Success-path + failure-mode breakdown */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Success Path Breakdown</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-stone-200 text-left text-xs uppercase tracking-wider text-stone-500 dark:border-stone-800 dark:text-stone-400">
                  <th className="py-2 pr-3">Path</th>
                  <th className="py-2 pr-3">Count</th>
                  <th className="py-2 pr-3">% Total</th>
                  <th className="py-2 pr-3">% Successes</th>
                </tr>
              </thead>
              <tbody>
                {successPaths.map((p) => (
                  <tr key={p.path} className="border-b border-stone-100 dark:border-stone-800">
                    <td className="py-2 pr-3 font-mono text-stone-900 dark:text-stone-100">{p.path}</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{formatNumber(p.count)}</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(p.pct_of_total)}%</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(p.pct_of_successes)}%</td>
                  </tr>
                ))}
                {successPaths.length === 0 && (
                  <tr><td colSpan={4} className="py-3 text-stone-500 dark:text-stone-400">No success-path data.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Failure Mode Breakdown</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-stone-200 text-left text-xs uppercase tracking-wider text-stone-500 dark:border-stone-800 dark:text-stone-400">
                  <th className="py-2 pr-3">Mode</th>
                  <th className="py-2 pr-3">Count</th>
                  <th className="py-2 pr-3">% Failures</th>
                  <th className="py-2 pr-3">% Total</th>
                </tr>
              </thead>
              <tbody>
                {failureModes.map((m) => (
                  <tr key={m.mode} className="border-b border-stone-100 dark:border-stone-800">
                    <td className="py-2 pr-3 font-mono text-stone-900 dark:text-stone-100">{m.mode}</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{formatNumber(m.count)}</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(m.pct_of_failures)}%</td>
                    <td className="py-2 pr-3 text-stone-600 dark:text-stone-400">{pctFmt.format(m.pct_of_total)}%</td>
                  </tr>
                ))}
                {failureModes.length === 0 && (
                  <tr><td colSpan={4} className="py-3 text-stone-500 dark:text-stone-400">No failure-mode data.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Run-config metadata */}
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Fallback Run Config</h3>
          <DetailRow label="Mutation Fallback Enabled" value={String(meta.mutation_fallback_enabled ?? 'n/a')} />
          <DetailRow label="Max Fallback Rounds" value={formatNumber(meta.max_fallback_rounds)} />
          <DetailRow label="Cooperative Seeding" value={String(meta.cooperative_seeding ?? 'n/a')} />
          <DetailRow label="Cooperative N (BoN cap)" value={formatNumber(meta.cooperative_n)} />
          <DetailRow label="Planner Temp Escalation" value={formatNumber(meta.planner_temp_escalation)} />
          <DetailRow label="Seed" value={formatNumber(meta.seed)} />
          <DetailRow label="Start Idx" value={formatNumber(meta.start_idx)} />
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Leak &amp; Recovery Rates</h3>
          <DetailRow label="GT Leak Rate" value={formatPct(summary?.gt_leak_rate)} />
          <DetailRow label="Defense Rate" value={formatPct(summary?.defense_rate)} />
          <DetailRow label="Extractor Recovery Rate" value={formatPct(summary?.extractor_recovery_rate)} />
          <DetailRow label="Total Success (exact)" value={formatNumber(summary?.total_success_exact)} />
          <DetailRow label="Total Success (extractor)" value={formatNumber(summary?.total_success_extractor)} />
          <DetailRow label="Variants Generated" value={formatNumber(diag?.variant_total)} />
        </div>
      </div>
    </Panel>
  );
}

function Th(props: { children: ReactNode }) {
  return <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide">{props.children}</th>;
}

function Td(props: { children: ReactNode; mono?: boolean; className?: string }) {
  return (
    <td className={`px-4 py-3 align-top text-sm ${props.className ?? ''} ${props.mono ? 'font-mono text-xs' : ''}`}>
      {props.children}
    </td>
  );
}
