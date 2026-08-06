import { useRunStore } from '../store/runStore';
import type { Attempt } from '../types/autored';
import { isRunSuccessful } from '../utils/success';

export default function TimelineSidebar() {
  const { selectedRun, selectedAttemptIndex, setSelectedAttempt } = useRunStore();
  if (!selectedRun) return null;
  const runSucceeded = isRunSuccessful(selectedRun);

  const getAttemptColor = (a: Attempt): string => {
    if (a.extractor_match) return 'bg-green-500';
    if (a.ground_truth_found && !a.extractor_match) return 'bg-red-500';
    if (a.ground_truth_found) return 'bg-yellow-500';
    if (a.judge.decision === 'ATTACK') return 'bg-blue-500';
    return 'bg-slate-400';
  };

  return (
    <div className="w-full bg-white flex flex-col h-full overflow-hidden">
      {/* Run Header */}
      <div className="p-4 border-b border-slate-200">
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Run</p>
        <p className="font-mono text-sm font-bold text-slate-900 mt-1 truncate">{selectedRun.experiment.run_id}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${runSucceeded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {runSucceeded ? 'SUCCESS' : 'FAILED'}
          </span>
          <span className="text-xs text-slate-500">{selectedRun.result.total_attempts} attempts</span>
        </div>
      </div>

      {/* Attempt List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {selectedRun.attempts.map((attempt: Attempt) => {
          const isSelected = attempt.attempt_number - 1 === selectedAttemptIndex;
          const color = getAttemptColor(attempt);
          const isStar = attempt.extractor_match || (attempt.ground_truth_found && attempt.attempt_number === selectedRun.result.total_attempts);

          return (
            <button
              key={attempt.attempt_number}
              onClick={() => setSelectedAttempt(attempt.attempt_number - 1)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                isSelected ? 'bg-slate-100 ring-1 ring-slate-300' : 'hover:bg-slate-50'
              }`}
            >
              <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-medium text-slate-700">Attempt {attempt.attempt_number}</span>
                  {isStar && <span className="text-yellow-500 text-xs">⭐</span>}
                </div>
                <p className="text-xs text-slate-500 truncate">{attempt.generator.strategy}</p>
              </div>
              {attempt.ground_truth_found && (
                <span className="text-xs text-amber-600 font-medium">leak</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Legend */}
      <div className="p-3 border-t border-slate-200 text-xs text-slate-500 space-y-1.5">
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500" /> Attack</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-yellow-500" /> Leak</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-green-500" /> Success</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-500" /> Extract Fail</div>
      </div>
    </div>
  );
}
