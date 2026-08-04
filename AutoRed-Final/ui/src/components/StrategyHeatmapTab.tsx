import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { useRunStore } from '../store/runStore';

const COLORS = ['#0d9488', '#6366f1', '#f59e0b', '#22c55e', '#ef4444', '#ec4899', '#06b6d4'];

export default function StrategyHeatmapTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;
  const strategyStats = selectedRun.strategy_stats;

  const strategySuccess: Record<string, { total: number; successes: number }> = {};
  attempts.forEach((a) => {
    const s = a.generator.strategy;
    if (!strategySuccess[s]) strategySuccess[s] = { total: 0, successes: 0 };
    strategySuccess[s].total += 1;
    if (a.ground_truth_found) strategySuccess[s].successes += 1;
  });

  const pieData = Object.entries(strategySuccess).map(([name, d]) => ({
    name,
    value: d.successes,
    total: d.total,
    rate: d.total > 0 ? ((d.successes / d.total) * 100).toFixed(1) : '0',
  }));

  const barData = Object.entries(strategyStats).map(([name, s]) => ({
    name,
    successes: s.successes,
    failures: s.failures,
    partialLeaks: s.partial_leaks,
  }));

  const judgeDist = attempts.reduce((acc, a) => {
    const d = a.judge.decision;
    acc[d] = (acc[d] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const judgePieData = Object.entries(judgeDist).map(([name, value]) => ({ name, value }));

  const tooltipStyle = {
    backgroundColor: 'var(--elevated)',
    borderColor: 'var(--border)',
    borderRadius: '0.5rem',
  };

  return (
    <div className="space-y-4 pb-8">
      <ChartCard title="Strategy Usage">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={3}
              dataKey="value"
            >
              {pieData.map((_, i) => (
                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-2 flex flex-wrap justify-center gap-2">
          {pieData.map((d, i) => (
            <span key={d.name} className="flex items-center gap-1 text-xs text-stone-700 dark:text-stone-300">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              {d.name} ({d.rate}%)
            </span>
          ))}
        </div>
      </ChartCard>

      <ChartCard title="Strategy Breakdown">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-stone-200 dark:text-stone-800" />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
            <YAxis tick={{ fontSize: 12, fill: 'currentColor' }} className="text-stone-600 dark:text-stone-400" />
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
            <Legend />
            <Bar dataKey="successes" stackId="a" fill="#22c55e" name="Successes" />
            <Bar dataKey="partialLeaks" stackId="a" fill="#f59e0b" name="Partial" />
            <Bar dataKey="failures" stackId="a" fill="#ef4444" name="Failures" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Judge Decision Distribution">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={judgePieData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={3}
              dataKey="value"
            >
              {judgePieData.map((d, i) => (
                <Cell key={`cell-${d.name}`} fill={i === 0 ? '#22c55e' : '#f59e0b'} />
              ))}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: 'currentColor' }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-2 flex flex-wrap justify-center gap-2">
          {judgePieData.map((d, i) => (
            <span key={d.name} className="flex items-center gap-1 text-xs text-stone-700 dark:text-stone-300">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: i === 0 ? '#22c55e' : '#f59e0b' }} />
              {d.name}: {d.value}
            </span>
          ))}
        </div>
      </ChartCard>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Strategy Details</h3>
        <div className="overflow-x-auto rounded-lg border border-stone-200 dark:border-stone-800">
          <table className="min-w-full text-sm">
            <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
              <tr>
                <th className="px-4 py-3">Strategy</th>
                <th className="px-4 py-3 text-center">Used</th>
                <th className="px-4 py-3 text-center">Successes</th>
                <th className="px-4 py-3 text-center">Failures</th>
                <th className="px-4 py-3 text-center">Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 bg-white dark:divide-stone-800 dark:bg-stone-900">
              {Object.entries(strategySuccess).map(([name, d]) => (
                <tr key={name}>
                  <td className="px-4 py-3 font-mono text-xs">{name}</td>
                  <td className="px-4 py-3 text-center text-stone-900 dark:text-stone-100">{d.total}</td>
                  <td className="px-4 py-3 text-center font-bold text-emerald-700 dark:text-emerald-400">{d.successes}</td>
                  <td className="px-4 py-3 text-center font-bold text-rose-700 dark:text-rose-400">{d.total - d.successes}</td>
                  <td className="px-4 py-3 text-center">
                    <span className={d.successes > 0 ? 'font-bold text-emerald-700 dark:text-emerald-400' : 'text-stone-500 dark:text-stone-400'}>
                      {d.total > 0 ? ((d.successes / d.total) * 100).toFixed(0) : 0}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
