import { Attempt } from '../types/autored';

export default function VictimCard({ attempt, accessCode }: { attempt: Attempt; accessCode?: string }) {
  const response = attempt.victim.raw_output;

  const renderHighlightedResponse = () => {
    if (!accessCode || !response) return <p className="whitespace-pre-wrap text-sm text-stone-700 dark:text-stone-300">{response}</p>;
    const escaped = accessCode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const parts = response.split(new RegExp(`(${escaped})`, 'gi'));
    return (
      <p className="whitespace-pre-wrap text-sm text-stone-700 dark:text-stone-300">
        {parts.map((part, i) =>
          part.toLowerCase() === accessCode.toLowerCase() ? (
            <mark key={i} className="rounded bg-amber-200 px-1 font-bold text-amber-900 dark:bg-amber-600/40 dark:text-amber-100">{part}</mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </p>
    );
  };

  return (
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-stone-100 dark:bg-stone-800">🦙</span>
          Victim Response
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-stone-500 dark:text-stone-400">{attempt.victim.output_length} chars</span>
          <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${attempt.ground_truth_found ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
            GT Found: {attempt.ground_truth_found ? 'YES' : 'NO'}
          </span>
        </div>
      </div>

      <div className="max-h-80 overflow-y-auto rounded-lg border border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-950">
        {renderHighlightedResponse()}
      </div>
    </div>
  );
}
