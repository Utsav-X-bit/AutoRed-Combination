import { useRunStore } from '../store/runStore';

export default function AttackEvolutionTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const attempts = selectedRun.attempts;
  const strategies = attempts.map((a) => a.generator.strategy);
  const uniqueStrategies = [...new Set(strategies)];

  const strategyPerf = uniqueStrategies.map((s) => {
    const sAttempts = attempts.filter((a) => a.generator.strategy === s);
    return {
      strategy: s,
      count: sAttempts.length,
      successes: sAttempts.filter((a) => a.ground_truth_found).length,
      leaks: sAttempts.filter((a) => a.extractor_match).length,
      avgTokens: Math.round(sAttempts.reduce((sum, a) => sum + a.generator.output_tokens, 0) / sAttempts.length),
    };
  });

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
    <div className="space-y-4 pb-8">
      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Strategy Timeline</h3>
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {attempts.map((a, i) => (
            <div key={a.attempt_number} className="flex flex-shrink-0 flex-col items-center">
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold ${
                  a.ground_truth_found
                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
                    : a.extractor_match
                    ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400'
                    : 'bg-stone-200 text-stone-600 dark:bg-stone-800 dark:text-stone-400'
                }`}
                title={`Attempt ${a.attempt_number}: ${a.generator.strategy}`}
              >
                {a.attempt_number}
              </div>
              <span className="mt-1 max-w-[64px] truncate text-[10px] text-stone-500 dark:text-stone-400">
                {a.generator.strategy.slice(0, 6)}
              </span>
              {i < attempts.length - 1 && (
                <span className="mt-1 text-xs text-stone-300 dark:text-stone-700">→</span>
              )}
            </div>
          ))}
        </div>
      </section>

      {changes.length > 0 && (
        <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
          <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Strategy Changes</h3>
          <div className="space-y-2">
            {changes.map((c, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="w-16 text-xs text-stone-500 dark:text-stone-400">Attempt {c.at}</span>
                <span className="rounded bg-stone-200 px-2 py-0.5 font-mono text-xs text-stone-700 dark:bg-stone-800 dark:text-stone-300">{c.from}</span>
                <span className="text-stone-400 dark:text-stone-500">→</span>
                <span className="rounded bg-indigo-100 px-2 py-0.5 font-mono text-xs text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-400">{c.to}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Strategy Performance</h3>
        <div className="overflow-x-auto rounded-lg border border-stone-200 dark:border-stone-800">
          <table className="min-w-full text-sm">
            <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
              <tr>
                <th className="px-4 py-3">Strategy</th>
                <th className="px-4 py-3 text-center">Used</th>
                <th className="px-4 py-3 text-center">Successes</th>
                <th className="px-4 py-3 text-center">Extracted</th>
                <th className="px-4 py-3 text-center">Avg Tokens</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 bg-white dark:divide-stone-800 dark:bg-stone-900">
              {strategyPerf.map((s) => (
                <tr key={s.strategy}>
                  <td className="px-4 py-3 font-mono text-xs">{s.strategy}</td>
                  <td className="px-4 py-3 text-center text-stone-900 dark:text-stone-100">{s.count}</td>
                  <td className="px-4 py-3 text-center font-bold text-emerald-700 dark:text-emerald-400">{s.successes}</td>
                  <td className="px-4 py-3 text-center font-bold text-amber-700 dark:text-amber-400">{s.leaks}</td>
                  <td className="px-4 py-3 text-center text-stone-900 dark:text-stone-100">{s.avgTokens}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {selectedRun.best_attack && (
        <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
          <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Best Attack</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-400">
                {selectedRun.best_attack.strategy}
              </span>
              <span className="text-xs text-stone-500 dark:text-stone-400">score: {selectedRun.best_attack.score}</span>
            </div>
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 dark:border-rose-900/50 dark:bg-rose-950/20">
              <p className="font-mono text-sm leading-relaxed text-rose-900 dark:text-rose-100">
                “{selectedRun.best_attack.prompt}”
              </p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
