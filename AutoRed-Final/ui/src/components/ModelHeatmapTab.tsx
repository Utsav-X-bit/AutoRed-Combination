import { useRunStore } from '../store/runStore';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export default function ModelHeatmapTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;

  const timingData = attempts.map((a) => ({
    attempt: a.attempt_number,
    time: a.attempt_time_ms,
    strategy: a.generator.strategy,
    success: a.ground_truth_found ? 1 : 0,
  }));

  const judgeData = attempts.map((a) => ({
    attempt: a.attempt_number,
    confidence: Math.round(a.judge.confidence * 100),
    decision: a.judge.decision,
  }));

  const extractorData = attempts.map((a) => {
    const best = a.extractor.ranked_candidates[0];
    return {
      attempt: a.attempt_number,
      score: best ? Math.round(best.score * 100) : 0,
      match: a.extractor_match ? 1 : 0,
    };
  });

  const strategyTiming: Record<string, { total: number; count: number }> = {};
  attempts.forEach((a) => {
    const s = a.generator.strategy;
    if (!strategyTiming[s]) strategyTiming[s] = { total: 0, count: 0 };
    strategyTiming[s].total += a.attempt_time_ms;
    strategyTiming[s].count += 1;
  });
  const strategyAvgTiming = Object.entries(strategyTiming).map(([name, d]) => ({
    name,
    avgMs: Math.round(d.total / d.count),
  }));

  const tooltipStyle = {
    backgroundColor: 'var(--elevated)',
    borderColor: 'var(--border)',
    borderRadius: '0.5rem',
  };

  const gridColor = 'currentColor';
  const tickColor = 'currentColor';

  return (
    <div className="space-y-4 pb-8">
      <ChartCard title="Attempt Timing (ms)">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={timingData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Bar dataKey="time" fill="#f59e0b" name="Time (ms)">
              {timingData.map((entry, index) => (
                <Cell key={index} fill={entry.success ? '#22c55e' : '#f59e0b'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Judge Confidence (%)">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={judgeData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Bar dataKey="confidence" name="Confidence">
              {judgeData.map((entry, index) => (
                <Cell key={index} fill={entry.decision === 'ATTACK' ? '#22c55e' : '#f59e0b'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Extractor Best Score (%)">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={extractorData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Bar dataKey="score" name="Score">
              {extractorData.map((entry, index) => (
                <Cell key={index} fill={entry.match ? '#22c55e' : '#64748b'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Avg Time by Strategy">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={strategyAvgTiming} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis type="number" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 10, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Bar dataKey="avgMs" fill="#3b82f6" name="Avg (ms)">
              {strategyAvgTiming.map((_, index) => (
                <Cell key={index} fill={['#0d9488', '#6366f1', '#f59e0b', '#22c55e', '#ef4444'][index % 5]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Model Load Times</h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Object.entries(selectedRun.models).map(([name, model]) => (
            <div key={name} className="rounded-lg border border-stone-200 p-3 text-center dark:border-stone-800">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{name}</p>
              <p className="font-display text-lg font-bold text-stone-900 dark:text-stone-100">{model.load_time.toFixed(1)}s</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="h-80 rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">{title}</h3>
      <div className="h-56">{children}</div>
    </section>
  );
}
