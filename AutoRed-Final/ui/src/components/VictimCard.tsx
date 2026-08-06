import { Attempt } from '../types/autored';

export default function VictimCard({ attempt, accessCode }: { attempt: Attempt; accessCode?: string }) {
  const response = attempt.victim.raw_output;

  const renderHighlightedResponse = () => {
    if (!accessCode || !response) return <p className="text-sm text-slate-700 whitespace-pre-wrap">{response}</p>;
    const escaped = accessCode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const parts = response.split(new RegExp(`(${escaped})`, 'gi'));
    return (
      <p className="text-sm text-slate-700 whitespace-pre-wrap">
        {parts.map((part, i) =>
          part.toLowerCase() === accessCode.toLowerCase() ? (
            <mark key={i} className="bg-yellow-200 text-yellow-900 px-1 rounded font-bold">{part}</mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </p>
    );
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🦙</span> Victim Response
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">{attempt.victim.output_length} chars</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.ground_truth_found ? 'bg-yellow-100 text-yellow-700' : 'bg-slate-100 text-slate-600'}`}>
            GT Found: {attempt.ground_truth_found ? '✓ YES' : '✗ NO'}
          </span>
        </div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 max-h-64 overflow-y-auto">
        {renderHighlightedResponse()}
      </div>
    </div>
  );
}
