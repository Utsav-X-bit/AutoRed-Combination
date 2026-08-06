import { useRunStore } from '../store/runStore';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

export default function ModelHeatmapTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;

  // Timing heatmap data
  const timingData = attempts.map(a => ({
    attempt: a.attempt_number,
    time: a.attempt_time_ms,
    strategy: a.generator.strategy,
    success: a.ground_truth_found ? 1 : 0,
  }));

  // Judge confidence trend
  const judgeData = attempts.map(a => ({
    attempt: a.attempt_number,
    confidence: Math.round(a.judge.confidence * 100),
    decision: a.judge.decision,
  }));

  // Extractor score trend
  const extractorData = attempts.map(a => {
    const best = a.extractor.ranked_candidates[0];
    return {
      attempt: a.attempt_number,
      score: best ? Math.round(best.score * 100) : 0,
      match: a.extractor_match ? 1 : 0,
    };
  });

  // Avg timing by strategy
  const strategyTiming: Record<string, { total: number; count: number }> = {};
  attempts.forEach(a => {
    const s = a.generator.strategy;
    if (!strategyTiming[s]) strategyTiming[s] = { total: 0, count: 0 };
    strategyTiming[s].total += a.attempt_time_ms;
    strategyTiming[s].count += 1;
  });
  const strategyAvgTiming = Object.entries(strategyTiming).map(([name, d]) => ({
    name,
    avgMs: Math.round(d.total / d.count),
  }));

  return (
    <div className="space-y-4">
      {/* Attempt Timing */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Attempt Timing (ms)</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={timingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="time" fill="#f59e0b" name="Time (ms)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Judge Confidence */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Judge Confidence (%)</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={judgeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="confidence" name="Confidence">
                {judgeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.decision === 'ATTACK' ? '#22c55e' : '#eab308'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Extractor Score */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Extractor Best Score (%)</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={extractorData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="attempt" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="score" name="Score">
                {extractorData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.match ? '#22c55e' : '#64748b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Avg Time by Strategy */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Avg Time by Strategy</h3>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={strategyAvgTiming} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="avgMs" fill="#3b82f6" name="Avg (ms)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Load Times */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Model Load Times</h3>
        <div className="grid grid-cols-4 gap-3 text-center">
          {Object.entries(selectedRun.models).map(([name, model]) => (
            <div key={name} className="bg-slate-50 rounded-lg p-3">
              <p className="text-xs text-slate-500 capitalize">{name}</p>
              <p className="text-lg font-bold text-slate-900">{model.load_time.toFixed(1)}s</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
