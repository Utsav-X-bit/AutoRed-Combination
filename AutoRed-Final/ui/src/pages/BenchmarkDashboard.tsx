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
} from '../types/autored';

const numberFmt = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2,
});

const pctFmt = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const asDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const formatPct = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return '0.0%';
  return `${pctFmt.format(Number(value) * 100)}%`;
};

const formatDelta = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return 'n/a';
  const delta = Number(value);
  const prefix = delta > 0 ? '+' : '';
  return `${prefix}${pctFmt.format(delta * 100)} pp`;
};

const formatNumber = (value: number | undefined | null) => {
  if (!Number.isFinite(Number(value))) return 'n/a';
  return numberFmt.format(Number(value));
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

export default function BenchmarkDashboard() {
  const navigate = useNavigate();
  // Nested benchmark_ids are slash-form "<model>/<chars>". The router stores the
  // URL-encoded form ("%2F"); decode once here so comparisons against the raw
  // API value (decoded) match. Legacy single-segment ids decode to themselves.
  const { benchmarkId: rawBenchmarkId } = useParams<{ benchmarkId?: string }>();
  const benchmarkId = rawBenchmarkId ? decodeURIComponent(rawBenchmarkId) : undefined;
  const [benchmarks, setBenchmarks] = useState<BenchmarkListItem[]>([]);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string>('');
  const [benchmarkDetail, setBenchmarkDetail] = useState<BenchmarkDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runFilter, setRunFilter] = useState('');
  const [archiveFilter, setArchiveFilter] = useState<'all' | string>('all');

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
    return () => {
      cancelled = true;
    };
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
    return () => {
      cancelled = true;
    };
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
  const traceRuns = useMemo(
    () => sortTraceRuns(benchmarkDetail?.trace_runs ?? []),
    [benchmarkDetail],
  );

  const filteredTraceRuns = useMemo(() => {
    const query = runFilter.trim().toLowerCase();
    return traceRuns.filter((run) => {
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
  }, [archiveFilter, runFilter, traceRuns]);

  const currentSuccessRate = summary ? Number(summary.success_rate ?? 0) : 0;
  const currentVerifiedRate = summary && summary.total_rounds
    ? Number(summary.verified_success ?? 0) / Number(summary.total_rounds)
    : 0;
  const currentAvgAttempts = summary ? Number(summary.avg_attempts_on_success ?? 0) : 0;
  const currentTop1 = summary ? Number(summary.top1_success ?? 0) : 0;
  const currentTop3 = summary ? Number(summary.top3_success ?? 0) : 0;
  const currentTop5 = summary ? Number(summary.top5_success ?? 0) : 0;
  const currentExtractor = summary?.extractor_metrics ?? {};

  const previousSummary = previousBenchmark
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

  const currentVsPrevious = currentBenchmark && previousSummary
    ? {
        success_rate: currentSuccessRate - previousSummary.success_rate,
        verified_rate: currentVerifiedRate - previousSummary.verified_rate,
        avg_attempts: currentAvgAttempts - previousSummary.avg_attempts,
        top1: currentTop1 - previousSummary.top1,
        top3: currentTop3 - previousSummary.top3,
        top5: currentTop5 - previousSummary.top5,
        f1: Number(currentExtractor.f1 ?? 0) - previousSummary.f1,
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
    return traceArchives.map((archive) => ({
      name: archive.archive_id,
      value: archive.run_count,
    }));
  }, [traceArchives]);

  if (loadingList) {
    return <div className="p-8 text-center text-slate-500">Loading benchmark explorer...</div>;
  }

  if (!benchmarks.length) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-slate-500">No benchmark summaries found</p>
          <p className="text-sm text-slate-400 mt-2">
            Run a benchmark to populate results/benchmarks and the dated trace archives.
          </p>
          <button
            onClick={() => navigate('/runs')}
            className="mt-4 text-sm text-blue-600 hover:text-blue-700"
          >
            Back to Runs
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => navigate('/runs')}
              className="text-sm text-slate-500 hover:text-slate-900 transition-colors"
            >
              Runs
            </button>
            <span className="text-slate-300">|</span>
            <div className="min-w-0">
              <h1 className="text-xl font-bold text-slate-900">Benchmark Explorer</h1>
              <p className="text-xs text-slate-500">
                Summary folders, worker rollups, trace archives, and per-run drilldown
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-slate-500">
              Benchmark
              <select
                value={selectedBenchmarkId}
                onChange={(e) => {
                  const nextId = e.target.value;
                  setSelectedBenchmarkId(nextId);
                  navigate(nextId ? `/benchmarks/${encodeURIComponent(nextId)}` : '/benchmarks', { replace: false });
                }}
                className="ml-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
              >
                {sortBenchmarks(benchmarks).map((item) => (
                  <option key={item.benchmark_id} value={item.benchmark_id}>
                    {item.benchmark_id}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={() => navigate('/runs')}
              className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm font-medium transition-colors"
            >
              Run History
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <section className="grid grid-cols-2 lg:grid-cols-6 gap-4">
          <MetricCard
            label="Success Rate"
            value={formatPct(summary ? Number(summary.success_rate ?? 0) : 0)}
            subtext={`${formatNumber(summary?.total_successes ?? 0)}/${formatNumber(summary?.total_rounds ?? 0)} rounds`}
            delta={currentVsPrevious ? formatDelta(currentVsPrevious.success_rate) : 'baseline n/a'}
          />
          <MetricCard
            label="Verified Rate"
            value={formatPct(currentVerifiedRate)}
            subtext={`${formatNumber(summary?.verified_success ?? 0)} verified`}
            delta={currentVsPrevious ? formatDelta(currentVsPrevious.verified_rate) : 'baseline n/a'}
          />
          <MetricCard
            label="Avg Attempts"
            value={formatNumber(currentAvgAttempts)}
            subtext="on successful rounds"
            delta={currentVsPrevious ? `${currentVsPrevious.avg_attempts > 0 ? '+' : ''}${numberFmt.format(currentVsPrevious.avg_attempts)}` : 'baseline n/a'}
          />
          <MetricCard
            label="Top-1 / Top-3 / Top-5"
            value={`${formatNumber(currentTop1)} / ${formatNumber(currentTop3)} / ${formatNumber(currentTop5)}`}
            subtext="rounds recovered by rank"
            delta={currentVsPrevious ? `Δ F1 ${formatDelta(currentVsPrevious.f1)}` : 'baseline n/a'}
          />
          <MetricCard
            label="Extractor F1"
            value={formatPct(Number(currentExtractor.f1 ?? 0))}
            subtext={`Precision ${formatPct(Number(currentExtractor.precision ?? 0))} · Recall ${formatPct(Number(currentExtractor.recall ?? 0))}`}
            delta={currentVsPrevious ? formatDelta(currentVsPrevious.f1) : 'baseline n/a'}
          />
          <MetricCard
            label="Trace Runs"
            value={formatNumber(traceRuns.length)}
            subtext={`${traceArchives.length} archive${traceArchives.length === 1 ? '' : 's'}`}
            delta={loadingDetail ? 'refreshing' : 'ready'}
          />
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
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
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="worker" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="successRate" fill="#2563eb" name="Success Rate %" />
                    <Bar dataKey="rounds" fill="#94a3b8" name="Rounds" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-slate-500">Worker summaries are not available for this benchmark.</p>
              )}
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              {(benchmarkDetail?.worker_summaries ?? []).map((worker) => (
                <div key={worker.worker_id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-slate-900">Worker {worker.worker_id}</p>
                    <span className="text-sm font-mono text-slate-600">{formatPct(worker.success_rate)}</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    {worker.successes}/{worker.rounds} successful rounds
                  </p>
                </div>
              ))}
            </div>
          </Panel>
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <Panel title="Trace Archives" className="xl:col-span-1">
            <div className="space-y-3">
              {traceArchives.length ? traceArchives.map((archive) => (
                <button
                  key={archive.archive_id}
                  onClick={() => setArchiveFilter(archiveFilter === archive.archive_id ? 'all' : archive.archive_id)}
                  className={`w-full rounded-lg border p-3 text-left transition-colors ${
                    archiveFilter === archive.archive_id
                      ? 'border-blue-400 bg-blue-50'
                      : 'border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-medium text-slate-900">{archive.archive_id}</p>
                    <span className="text-xs text-slate-500">{archive.run_count} runs</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Success {formatPct(archive.success_rate)} · Verified {formatPct(archive.verified_rate)} · Avg attempts {formatNumber(archive.avg_attempts_on_success)}
                  </p>
                </button>
              )) : (
                <p className="text-sm text-slate-500">No trace archives found for this benchmark.</p>
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
                        <Cell
                          key={entry.name}
                          fill={['#2563eb', '#16a34a', '#f59e0b', '#7c3aed', '#0f766e'][index % 5]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-slate-500">No archive distribution data available.</p>
              )}
            </div>
          </Panel>
        </section>

        <section className="grid grid-cols-1 gap-6">
          <Panel title="Run-Level Drilldown">
            <div className="flex flex-col lg:flex-row lg:items-center gap-3 mb-4">
              <input
                value={runFilter}
                onChange={(e) => setRunFilter(e.target.value)}
                placeholder="Filter by run id, scenario, worker, access code, generator, or victim"
                className="w-full lg:flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              />
              <select
                value={archiveFilter}
                onChange={(e) => setArchiveFilter(e.target.value)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
              >
                <option value="all">All archives</option>
                {traceArchives.map((archive) => (
                  <option key={archive.archive_id} value={archive.archive_id}>
                    {archive.archive_id}
                  </option>
                ))}
              </select>
              <div className="text-sm text-slate-500 lg:ml-auto">
                {filteredTraceRuns.length}/{traceRuns.length} runs shown
              </div>
            </div>

            <div className="overflow-auto rounded-lg border border-slate-200 bg-white">
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-500">
                  <tr>
                    <Th>Run</Th>
                    <Th>Scenario</Th>
                    <Th>Stage</Th>
                    <Th>Worker</Th>
                    <Th>Attempts</Th>
                    <Th>Status</Th>
                    <Th>Access Code</Th>
                    <Th>Generator</Th>
                    <Th>Victim</Th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTraceRuns.map((run) => (
                    <tr
                      key={`${run.run_id}-${run.file_path}`}
                      onClick={() => navigate(`/run/${encodeURIComponent(run.run_id)}`)}
                      className="cursor-pointer border-t border-slate-100 hover:bg-slate-50"
                    >
                      <Td mono>{run.run_id}</Td>
                      <Td mono>{run.scenario_id || 'unknown'}</Td>
                      <Td>{run.file_path.split('/').slice(-2, -1)[0] || 'unknown'}</Td>
                      <Td>{run.worker_id ?? 'n/a'}</Td>
                      <Td>{run.attempt_count ?? run.total_attempts}</Td>
                      <Td>
                        <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${
                          run.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {run.verified_success ? 'verified' : run.success ? 'success' : 'failed'}
                        </span>
                      </Td>
                      <Td className="font-mono text-amber-700">{run.access_code || 'n/a'}</Td>
                      <Td>{run.generator || 'n/a'}</Td>
                      <Td>{run.victim || 'n/a'}</Td>
                    </tr>
                  ))}
                  {!filteredTraceRuns.length && (
                    <tr>
                      <td colSpan={9} className="px-4 py-8 text-center text-slate-500">
                        No trace runs match the current filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <Panel title="Current Summary Payload">
            <pre className="max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100">
              {JSON.stringify(summary, null, 2)}
            </pre>
          </Panel>
          <Panel title="Selected Benchmark Trace Payload">
            <pre className="max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100">
              {JSON.stringify(benchmarkDetail, null, 2)}
            </pre>
          </Panel>
        </section>

        {loadingDetail && (
          <div className="text-sm text-slate-500">Refreshing benchmark detail...</div>
        )}
      </main>
    </div>
  );
}

function Panel(
  props: { title: string; className?: string; children: ReactNode },
) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 ${props.className ?? ''}`}>
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="text-sm font-bold text-slate-900">{props.title}</h2>
      </div>
      {props.children}
    </section>
  );
}

function MetricCard(props: { label: string; value: string; subtext: string; delta: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-xs text-slate-500 mb-1">{props.label}</p>
      <p className="text-lg font-bold text-slate-900 break-words">{props.value}</p>
      <p className="text-xs text-slate-500 mt-1">{props.subtext}</p>
      <p className="text-xs text-slate-400 mt-2">{props.delta}</p>
    </div>
  );
}

function DetailRow(props: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="py-2 border-b border-slate-100 last:border-b-0">
      <p className="text-xs text-slate-500">{props.label}</p>
      <p className={`text-sm text-slate-900 ${props.mono ? 'font-mono break-all' : 'break-words'}`}>
        {props.value}
      </p>
    </div>
  );
}

function Th(props: { children: ReactNode }) {
  return <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide">{props.children}</th>;
}

function Td(props: { children: ReactNode; mono?: boolean; className?: string }) {
  return (
    <td className={`px-4 py-3 align-top ${props.className ?? ''} ${props.mono ? 'font-mono text-xs' : ''}`}>
      {props.children}
    </td>
  );
}
