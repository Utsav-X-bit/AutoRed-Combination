import { useRunStore } from '../store/runStore';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area,
} from 'recharts';

export default function TokenAnalyticsTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;

  // Token data per attempt
  const tokenData = attempts.map(a => ({
    attempt: a.attempt_number,
    inputTokens: a.generator.input_tokens,
    outputTokens: a.generator.output_tokens,
    totalTokens: a.generator.input_tokens + a.generator.output_tokens,
  }));

  // Cumulative tokens
  let cumInput = 0, cumOutput = 0;
  const cumulativeData = tokenData.map(d => {
    cumInput += d.inputTokens;
    cumOutput += d.outputTokens;
    return { attempt: d.attempt, cumInput, cumOutput };
  });

  // Total stats
  const totalInput = tokenData.reduce((s, d) => s + d.inputTokens, 0);
  const totalOutput = tokenData.reduce((s, d) => s + d.outputTokens, 0);
  const avgInput = Math.round(totalInput / attempts.length);
  const avgOutput = Math.round(totalOutput / attempts.length);

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-white rounded-xl border border-slate-200 p-3 text-center">
          <p className="text-xs text-slate-500">Total Input</p>
          <p className="text-xl font-bold text-blue-600">{totalInput.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 text-center">
          <p className="text-xs text-slate-500">Total Output</p>
          <p className="text-xl font-bold text-purple-600">{totalOutput.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 text-center">
          <p className="text-xs text-slate-500">Avg Input</p>
          <p className="text-xl font-bold text-blue-600">{avgInput.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-3 text-center">
          <p className="text-xs text-slate-500">Avg Output</p>
          <p className="text-xl font-bold text-purple-600">{avgOutput.toLocaleString()}</p>
        </div>
      </div>

      {/* Per-Attempt Tokens */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Tokens Per Attempt</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tokenData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="inputTokens" fill="#3b82f6" name="Input" />
              <Bar dataKey="outputTokens" fill="#8b5cf6" name="Output" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Cumulative Tokens */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Cumulative Tokens</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={cumulativeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="cumInput" stackId="1" stroke="#3b82f6" fill="#3b82f6" name="Input" />
              <Area type="monotone" dataKey="cumOutput" stackId="2" stroke="#8b5cf6" fill="#8b5cf6" name="Output" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Token Trend */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Token Trend</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={tokenData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="totalTokens" stroke="#f59e0b" strokeWidth={2} name="Total" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
