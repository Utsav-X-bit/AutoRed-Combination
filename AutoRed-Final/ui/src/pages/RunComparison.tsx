import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  BarChart,
  Bar,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AutoRedRun, StrategyStat } from '../types/autored';
import { normalizeRun } from '../utils/normalizeRun';

const numberFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

const getMetric = (run: AutoRedRun, key: string) => {
  switch (key) {
    case 'success':
      return run.result.ground_truth_success ? 1 : 0;
    case 'extractor':
      return run.result.extractor_success ? 1 : 0;
    case 'attempts':
      return run.attempts.length || run.result.total_attempts || 0;
    case 'runtime':
      return run.timing.total_run_time;
    default:
      return 0;
  }
};

export default function RunComparison() {
  const { runIdA, runIdB } = useParams<{ runIdA: string; runIdB: string }>();
  const navigate = useNavigate();
  const [runA, setRunA] = useState<AutoRedRun | null>(null);
  const [runB, setRunB] = useState<AutoRedRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runIdA || !runIdB) {
      setError('Two run ids are required to compare.');
      setLoading(false);
      return;
    }
    Promise.all([
      fetch(`/api/run/${encodeURIComponent(runIdA)}`).then((r) => {
        if (!r.ok) throw new Error(`Run ${runIdA}: ${r.status}`);
        return r.json();
      }),
      fetch(`/api/run/${encodeURIComponent(runIdB)}`).then((r) => {
        if (!r.ok) throw new Error(`Run ${runIdB}: ${r.status}`);
        return r.json();
      }),
    ])
      .then(([a, b]) => {
        setRunA(normalizeRun(a));
        setRunB(normalizeRun(b));
      })
      .catch((err) => {
        console.error('Failed to load runs:', err);
        setError(err instanceof Error ? err.message : 'Failed to load runs');
      })
      .finally(() => setLoading(false));
  }, [runIdA, runIdB]);

  if (loading) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center text-stone-500 dark:text-stone-400">
        <div className="flex items-center gap-3">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-300 border-t-teal-600 dark:border-stone-700 dark:border-t-teal-500" />
          Loading runs…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <main className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-7xl items-center justify-center p-6">
        <div className="text-center">
          <p className="text-rose-600 dark:text-rose-400">{error}</p>
          <button onClick={() => navigate('/runs')} className="mt-4 text-teal-700 hover:underline dark:text-teal-400">
            Back to runs
          </button>
        </div>
      </main>
    );
  }

  if (!runA || !runB) return null;

  const allStrategies = useMemo(
    () =>
      [...new Set([...Object.keys(runA.strategy_stats), ...Object.keys(runB.strategy_stats)])].sort(),
    [runA, runB],
  );

  const metricData = useMemo(() => {
    const keys = [
      { key: 'success', label: 'Success', formatter: (v: number) => (v ? 'Yes' : 'No') },
      { key: 'extractor', label: 'Extractor', formatter: (v: number) => (v ? 'Yes' : 'No') },
      { key: 'attempts', label: 'Attempts', formatter: numberFmt.format },
      { key: 'runtime', label: 'Runtime (s)', formatter: numberFmt.format },
    ] as const;
    return keys.map((m) => ({
      label: m.label,
      runA: getMetric(runA, m.key),
      runB: getMetric(runB, m.key),
      formatter: m.formatter,
    }));
  }, [runA, runB]);

  return (
    <main className="mx-auto max-w-[1600px] p-4 lg:p-6">
      <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
            Run Comparison
          </h1>
          <p className="text-sm text-stone-500 dark:text-stone-400">Side-by-side metrics, strategy breakdown, and attempt timeline.</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => navigate('/runs')} className="btn-default">Back to runs</button>
          <button onClick={() => navigate('/')} className="btn-default">Dashboard</button>
        </div>
      </header>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <RunSummaryCard label="Run A" run={runA} />
        <RunSummaryCard label="Run B" run={runB} />
      </section>

      <section className="mt-6">
        <Panel title="Metric Comparison">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricData}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-stone-200 dark:text-stone-800" />
                <XAxis dataKey="label" tick={{ fontSize: 12, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                <YAxis tick={{ fontSize: 12, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--elevated)', borderColor: 'var(--border)', borderRadius: '0.5rem' }}
                  itemStyle={{ color: 'currentColor' }}
                  formatter={(value: number, _name: string, props: { payload?: { formatter?: (v: number) => string } }) => {
                    const formatter = props?.payload?.formatter ?? numberFmt.format;
                    return [formatter(value), _name];
                  }}
                />
                <Legend />
                <Bar dataKey="runA" fill="#0d9488" name={`Run A: ${runA.experiment.run_id}`} />
                <Bar dataKey="runB" fill="#6366f1" name={`Run B: ${runB.experiment.run_id}`} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </section>

      <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Metadata">
          <ComparisonRow label="Timestamp" a={runA.experiment.timestamp} b={runB.experiment.timestamp} />
          <ComparisonRow label="Scenario ID" a={runA.experiment.scenario_id} b={runB.experiment.scenario_id} />
          <ComparisonRow label="Seed" a={String(runA.experiment.seed)} b={String(runB.experiment.seed)} />
          <ComparisonRow label="Benchmark Mode" a={runA.experiment.benchmark_mode ? 'YES' : 'NO'} b={runB.experiment.benchmark_mode ? 'YES' : 'NO'} />
          <ComparisonRow label="Max Attempts" a={String(runA.experiment.max_attempts)} b={String(runB.experiment.max_attempts)} />
          <ComparisonRow label="Generator" a={runA.models.generator.name} b={runB.models.generator.name} />
          <ComparisonRow label="Victim" a={runA.models.victim.name} b={runB.models.victim.name} />
        </Panel>

        <Panel title="Timing">
          <ComparisonRow label="Total Run Time" a={`${numberFmt.format(runA.timing.total_run_time)}s`} b={`${numberFmt.format(runB.timing.total_run_time)}s`} />
          <ComparisonRow label="Avg Attempt Time" a={`${numberFmt.format(runA.timing.average_attempt_time)}s`} b={`${numberFmt.format(runB.timing.average_attempt_time)}s`} />
          <ComparisonRow label="Model Loading Time" a={`${numberFmt.format(runA.timing.model_loading_time)}s`} b={`${numberFmt.format(runB.timing.model_loading_time)}s`} />
        </Panel>
      </section>

      <section className="mt-6">
        <Panel title="Strategy Comparison">
          <div className="overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
            <table className="min-w-full text-sm">
              <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
                <tr>
                  <Th>Strategy</Th>
                  <Th className="text-right">Run A Success</Th>
                  <Th className="text-right">Run A Leak</Th>
                  <Th className="text-right">Run A Fail</Th>
                  <Th className="text-right">Run B Success</Th>
                  <Th className="text-right">Run B Leak</Th>
                  <Th className="text-right">Run B Fail</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 bg-white dark:divide-stone-800 dark:bg-stone-900">
                {allStrategies.map((strategy) => {
                  const a = runA.strategy_stats[strategy] as StrategyStat | undefined;
                  const b = runB.strategy_stats[strategy] as StrategyStat | undefined;
                  return (
                    <tr key={strategy}>
                      <Td mono>{strategy}</Td>
                      <Td className="text-right font-semibold text-emerald-700 dark:text-emerald-400">{a?.successes ?? 0}</Td>
                      <Td className="text-right text-amber-700 dark:text-amber-400">{a?.partial_leaks ?? 0}</Td>
                      <Td className="text-right text-rose-700 dark:text-rose-400">{a?.failures ?? 0}</Td>
                      <Td className="text-right font-semibold text-emerald-700 dark:text-emerald-400">{b?.successes ?? 0}</Td>
                      <Td className="text-right text-amber-700 dark:text-amber-400">{b?.partial_leaks ?? 0}</Td>
                      <Td className="text-right text-rose-700 dark:text-rose-400">{b?.failures ?? 0}</Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </section>

      <section className="mt-6">
        <Panel title="Per-Attempt Comparison">
          <div className="overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
            <table className="min-w-full text-sm">
              <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
                <tr>
                  <Th>#</Th>
                  <Th>Run A Strategy</Th>
                  <Th className="text-center">Run A Result</Th>
                  <Th>Run B Strategy</Th>
                  <Th className="text-center">Run B Result</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 bg-white dark:divide-stone-800 dark:bg-stone-900">
                {Array.from({ length: Math.max(runA.attempts.length, runB.attempts.length) }).map((_, i) => {
                  const a = runA.attempts[i];
                  const b = runB.attempts[i];
                  return (
                    <tr key={i}>
                      <Td mono>{i + 1}</Td>
                      <Td>{a ? <span className="font-mono text-xs">{a.generator.strategy}</span> : <Empty />}</Td>
                      <Td className="text-center">{a ? <Result result={a.ground_truth_found} extractor={a.extractor_match} /> : <Empty />}</Td>
                      <Td>{b ? <span className="font-mono text-xs">{b.generator.strategy}</span> : <Empty />}</Td>
                      <Td className="text-center">{b ? <Result result={b.ground_truth_found} extractor={b.extractor_match} /> : <Empty />}</Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </section>

      <section className="mt-6 flex justify-end gap-3">
        <button onClick={() => navigate(`/run/${runA.experiment.run_id}`)} className="btn-primary">View Run A Detail</button>
        <button onClick={() => navigate(`/run/${runB.experiment.run_id}`)} className="btn-primary">View Run B Detail</button>
      </section>
    </main>
  );
}

function RunSummaryCard({ label, run }: { label: string; run: AutoRedRun }) {
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-display text-sm font-semibold text-stone-900 dark:text-stone-100">{label}</h2>
        <span className="font-mono text-xs text-stone-500 dark:text-stone-400">{run.experiment.run_id}</span>
      </div>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        <MiniStat label="Success" value={run.result.ground_truth_success ? 'Yes' : 'No'} positive={run.result.ground_truth_success} />
        <MiniStat label="Verified" value={run.result.verified_success ? 'Yes' : 'No'} positive={run.result.verified_success} />
        <MiniStat label="Extractor" value={run.result.extractor_success ? 'Yes' : 'No'} positive={run.result.extractor_success} />
        <MiniStat label="Attempts" value={String(run.result.total_attempts)} />
        <MiniStat label="Generator" value={run.models.generator.name.split('/').pop()} mono />
        <MiniStat label="Victim" value={run.models.victim.name.split('/').pop()} mono />
      </div>
    </div>
  );
}

function MiniStat({ label, value, positive, mono }: { label: string; value?: string; positive?: boolean; mono?: boolean }) {
  const valueClass =
    positive === false
      ? 'text-rose-600 dark:text-rose-400'
      : positive === true
      ? 'text-emerald-600 dark:text-emerald-400'
      : 'text-stone-900 dark:text-stone-100';
  return (
    <div className="rounded-lg border border-stone-100 p-2.5 dark:border-stone-800">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${valueClass} ${mono ? 'break-all font-mono text-xs leading-5' : 'truncate'}`}>{value ?? 'n/a'}</p>
    </div>
  );
}

function Panel({ title, className, children }: { title: string; className?: string; children: ReactNode }) {
  return (
    <section className={`rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900 ${className ?? ''}`}>
      <h2 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">{title}</h2>
      {children}
    </section>
  );
}

function ComparisonRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div className="grid grid-cols-3 items-start gap-3 border-b border-stone-100 py-2 last:border-b-0 dark:border-stone-800">
      <dt className="text-xs font-medium text-stone-500 dark:text-stone-400">{label}</dt>
      <dd className="break-words text-sm text-stone-900 dark:text-stone-100">{a}</dd>
      <dd className="break-words text-sm text-stone-900 dark:text-stone-100">{b}</dd>
    </div>
  );
}

function Result({ result, extractor }: { result?: boolean; extractor?: boolean }) {
  if (result) {
    return <span className="inline-flex rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">ground truth</span>;
  }
  if (extractor) {
    return <span className="inline-flex rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-700 dark:bg-amber-950/60 dark:text-amber-400">extractor only</span>;
  }
  return <span className="inline-flex rounded-full bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-700 dark:bg-rose-950/60 dark:text-rose-400">fail</span>;
}

function Empty() {
  return <span className="text-stone-300 dark:text-stone-600">—</span>;
}

function Th({ children, className }: { children: ReactNode; className?: string }) {
  return <th className={`px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide ${className ?? ''}`}>{children}</th>;
}

function Td({ children, mono, className }: { children: ReactNode; mono?: boolean; className?: string }) {
  return <td className={`px-4 py-3 align-top text-sm ${className ?? ''} ${mono ? 'font-mono text-xs' : ''}`}>{children}</td>;
}
