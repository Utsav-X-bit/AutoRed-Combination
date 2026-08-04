import { useRunStore } from '../store/runStore';
import type { Attempt } from '../types/autored';
import { isRunSuccessful } from '../utils/success';

export default function TimelineSidebar() {
  const { selectedRun, selectedAttemptIndex, setSelectedAttempt } = useRunStore();
  if (!selectedRun) return null;
  const runSucceeded = isRunSuccessful(selectedRun);

  const getAttemptColor = (a: Attempt): string => {
    if (a.extractor_match) return 'bg-emerald-500';
    if (a.ground_truth_found && !a.extractor_match) return 'bg-rose-500';
    if (a.ground_truth_found) return 'bg-amber-500';
    if (a.judge.decision === 'ATTACK') return 'bg-blue-500';
    return 'bg-stone-400';
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden border-r border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
      <div className="border-b border-stone-200 p-4 dark:border-stone-800">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Run</p>
        <p className="mt-1 truncate font-mono text-sm font-semibold text-stone-900 dark:text-stone-100">{selectedRun.experiment.run_id}</p>
        <div className="mt-2 flex items-center gap-2">
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
              runSucceeded
                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
                : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'
            }`}
          >
            {runSucceeded ? 'SUCCESS' : 'FAILED'}
          </span>
          <span className="text-xs text-stone-500 dark:text-stone-400">{selectedRun.result.total_attempts} attempts</span>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto p-2">
        {selectedRun.attempts.map((attempt: Attempt) => {
          const isSelected = attempt.attempt_number - 1 === selectedAttemptIndex;
          const color = getAttemptColor(attempt);
          const isStar = attempt.extractor_match || (attempt.ground_truth_found && attempt.attempt_number === selectedRun.result.total_attempts);

          return (
            <button
              key={attempt.attempt_number}
              onClick={() => setSelectedAttempt(attempt.attempt_number - 1)}
              className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all ${
                isSelected
                  ? 'bg-stone-100 ring-1 ring-stone-300 dark:bg-stone-800 dark:ring-stone-700'
                  : 'hover:bg-stone-50 dark:hover:bg-stone-800/50'
              }`}
            >
              <span className={`h-2.5 w-2.5 flex-shrink-0 rounded-full ${color}`} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-medium text-stone-700 dark:text-stone-200">Attempt {attempt.attempt_number}</span>
                  {isStar && <span className="text-xs">⭐</span>}
                </div>
                <p className="truncate text-xs text-stone-500 dark:text-stone-400">{attempt.generator.strategy}</p>
              </div>
              {attempt.ground_truth_found && (
                <span className="text-xs font-medium text-amber-600 dark:text-amber-400">leak</span>
              )}
            </button>
          );
        })}
      </div>

      <div className="border-t border-stone-200 p-3 text-xs text-stone-500 dark:border-stone-800 dark:text-stone-400">
        <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-blue-500" /> Attack</div>
        <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-amber-500" /> Leak</div>
        <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-emerald-500" /> Success</div>
        <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-rose-500" /> Extract Fail</div>
      </div>
    </div>
  );
}
