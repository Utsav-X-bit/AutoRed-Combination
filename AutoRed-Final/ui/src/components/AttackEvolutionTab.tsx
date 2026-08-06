import { useRunStore } from '../store/runStore';

export default function AttackEvolutionTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;
  const strategies = attempts.map(a => a.generator.strategy);
  const uniqueStrategies = [...new Set(strategies)];

  // Strategy performance
  const strategyPerf = uniqueStrategies.map(s => {
    const sAttempts = attempts.filter(a => a.generator.strategy === s);
    return {
      strategy: s,
      count: sAttempts.length,
      successes: sAttempts.filter(a => a.ground_truth_found).length,
      leaks: sAttempts.filter(a => a.extractor_match).length,
      avgTokens: Math.round(sAttempts.reduce((sum, a) => sum + a.generator.output_tokens, 0) / sAttempts.length),
    };
  });

  // Strategy change points
  const changes: { from: string; to: string; at: number }[] = [];
  for (let i = 1; i < attempts.length; i++) {
    if (attempts[i].generator.strategy !== attempts[i - 1].generator.strategy) {
      changes.push({
        from: attempts[i - 1].generator.strategy,
        to: attempts[i].generator.strategy,
        at: attempts[i].attempt_number,
      });
    }
  }

  return (
    <div className="space-y-4">
      {/* Strategy Timeline */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Timeline</h3>
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {attempts.map((a, i) => (
            <div key={a.attempt_number} className="flex-shrink-0 flex flex-col items-center">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                  a.ground_truth_found
                    ? 'bg-green-100 text-green-700'
                    : a.extractor_match
                    ? 'bg-green-100 text-green-700'
                    : 'bg-slate-100 text-slate-600'
                }`}
                title={`Attempt ${a.attempt_number}: ${a.generator.strategy}`}
              >
                {a.attempt_number}
              </div>
              <span className="text-[10px] text-slate-500 mt-1 truncate max-w-[60px]">
                {a.generator.strategy.slice(0, 6)}
              </span>
              {i < attempts.length - 1 && (
                <span className="text-slate-300 text-xs mt-1">→</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Strategy Changes */}
      {changes.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Changes</h3>
          <div className="space-y-2">
            {changes.map((c, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="text-xs text-slate-400 w-16">Attempt {c.at}</span>
                <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-mono">{c.from}</span>
                <span className="text-slate-400">→</span>
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs font-mono">{c.to}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strategy Performance */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Strategy Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 px-3 text-xs font-medium text-slate-500">Strategy</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Used</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Leaks</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Extracted</th>
                <th className="text-center py-2 px-3 text-xs font-medium text-slate-500">Avg Tokens</th>
              </tr>
            </thead>
            <tbody>
              {strategyPerf.map((s) => (
                <tr key={s.strategy} className="border-b border-slate-100">
                  <td className="py-2 px-3 font-mono text-xs">{s.strategy}</td>
                  <td className="py-2 px-3 text-center">{s.count}</td>
                  <td className="py-2 px-3 text-center">
                    <span className={`font-bold ${s.successes > 0 ? 'text-green-600' : 'text-slate-400'}`}>
                      {s.successes}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span className={`font-bold ${s.leaks > 0 ? 'text-green-600' : 'text-slate-400'}`}>
                      {s.leaks}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center">{s.avgTokens}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Best Attack */}
      {selectedRun.best_attack && (
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-bold text-slate-900 mb-3">Best Attack</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
                {selectedRun.best_attack.strategy}
              </span>
              <span className="text-xs text-slate-500">score: {selectedRun.best_attack.score}</span>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-900 font-mono leading-relaxed">
                "{selectedRun.best_attack.prompt}"
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
