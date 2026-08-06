import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { useRunStore } from '../store/runStore';

const COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444', '#ec4899', '#06b6d4'];

export default function StrategyHeatmapTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;
  const strategyStats = selectedRun.strategy_stats;

  // Strategy success rate computed from attempts
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

  // Judge decision distribution
  const judgeDist = attempts.reduce((acc, a) => {
    const d = a.judge.decision;
    acc[d] = (acc[d] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const judgePieData = Object.entries(judgeDist).map(([name, value]) => ({ name, value }));

  return (
    <div className="space-y-4">
      {/* Strategy Distribution */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Usage</h3>
        <div className="h-48">
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
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-2 justify-center mt-2">
          {pieData.map((d, i) => (
            <span key={d.name} className="flex items-center gap-1 text-xs">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: COLORS[i % COLORS.length] }}
              />
              {d.name} ({d.rate}%)
            </span>
          ))}
        </div>
      </div>

      {/* Strategy Success/Failure */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Breakdown</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="successes" stackId="a" fill="#22c55e" name="Successes" />
              <Bar dataKey="partialLeaks" stackId="a" fill="#f59e0b" name="Partial" />
              <Bar dataKey="failures" stackId="a" fill="#ef4444" name="Failures" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Judge Decision Distribution */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Judge Decision Distribution</h3>
        <div className="h-48">
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
                {judgePieData.map((_, i) => (
                  <Cell key={`cell-${i}`} fill={i === 0 ? '#22c55e' : '#eab308'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex flex-wrap gap-2 justify-center mt-2">
          {judgePieData.map((d, i) => (
            <span key={d.name} className="flex items-center gap-1 text-xs">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: i === 0 ? '#22c55e' : '#eab308' }}
              />
              {d.name}: {d.value}
            </span>
          ))}
        </div>
      </div>

      {/* Strategy Details Table */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Strategy</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Used</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Successes</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Failures</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(strategySuccess).map(([name, d]) => (
                <tr key={name} className="border-b border-slate-100">
                  <td className="py-2 px-3 font-mono text-xs">{name}</td>
                  <td className="py-2 px-3 text-center">{d.total}</td>
                  <td className="py-2 px-3 text-center text-green-600 font-bold">{d.successes}</td>
                  <td className="py-2 px-3 text-center text-red-600 font-bold">{d.total - d.successes}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={`font-bold ${d.successes > 0 ? 'text-green-600' : 'text-slate-400'}`}>
                      {d.total > 0 ? ((d.successes / d.total) * 100).toFixed(0) : 0}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
