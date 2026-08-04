import { useRunStore } from '../store/runStore';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
} from 'recharts';

export default function TokenAnalyticsTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;

  const tokenData = attempts.map((a) => ({
    attempt: a.attempt_number,
    inputTokens: a.generator.input_tokens,
    outputTokens: a.generator.output_tokens,
    totalTokens: a.generator.input_tokens + a.generator.output_tokens,
  }));

  let cumInput = 0;
  let cumOutput = 0;
  const cumulativeData = tokenData.map((d) => {
    cumInput += d.inputTokens;
    cumOutput += d.outputTokens;
    return { attempt: d.attempt, cumInput, cumOutput };
  });

  const totalInput = tokenData.reduce((s, d) => s + d.inputTokens, 0);
  const totalOutput = tokenData.reduce((s, d) => s + d.outputTokens, 0);
  const avgInput = Math.round(totalInput / attempts.length);
  const avgOutput = Math.round(totalOutput / attempts.length);

  const tooltipStyle = {
    backgroundColor: 'var(--elevated)',
    borderColor: 'var(--border)',
    borderRadius: '0.5rem',
  };

  const gridColor = 'currentColor';
  const tickColor = 'currentColor';

  return (
    <div className="space-y-4 pb-8">
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Total Input" value={totalInput.toLocaleString()} accent="teal" />
        <StatCard label="Total Output" value={totalOutput.toLocaleString()} accent="indigo" />
        <StatCard label="Avg Input" value={avgInput.toLocaleString()} accent="teal" />
        <StatCard label="Avg Output" value={avgOutput.toLocaleString()} accent="indigo" />
      </section>

      <ChartCard title="Tokens Per Attempt">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={tokenData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Bar dataKey="inputTokens" fill="#0d9488" name="Input" />
            <Bar dataKey="outputTokens" fill="#6366f1" name="Output" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Cumulative Tokens">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={cumulativeData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Area type="monotone" dataKey="cumInput" stackId="1" stroke="#0d9488" fill="#0d9488" name="Input" />
            <Area type="monotone" dataKey="cumOutput" stackId="2" stroke="#6366f1" fill="#6366f1" name="Output" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Token Trend">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={tokenData}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridColor} className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="attempt" tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <YAxis tick={{ fontSize: 12, fill: tickColor }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Line type="monotone" dataKey="totalTokens" stroke="#f59e0b" strokeWidth={2} name="Total" />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent: 'teal' | 'indigo' }) {
  const valueClass = accent === 'teal' ? 'text-teal-700 dark:text-teal-400' : 'text-indigo-700 dark:text-indigo-400';
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-3 text-center shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className={`font-display text-xl font-bold ${valueClass}`}>{value}</p>
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
