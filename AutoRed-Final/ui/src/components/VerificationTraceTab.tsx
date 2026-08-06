import { useRunStore } from '../store/runStore';
import type { VerificationTraceItem } from '../types/autored';

function TraceBadges({ trace }: { trace: VerificationTraceItem }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
        trace.accepted_by_victim ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-600'
      }`}>
        victim accepted: {trace.accepted_by_victim ? 'yes' : 'no'}
      </span>
      <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
        trace.complete_match ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
      }`}>
        complete match: {trace.complete_match ? 'yes' : 'no'}
      </span>
    </div>
  );
}

export default function VerificationTraceTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const totalVerified = selectedRun.attempts.filter(
    a => a.extractor.verification_traces?.some(t => t.success)
  ).length;

  const totalFailed = selectedRun.attempts.filter(
    a => a.extractor.verification_traces && a.extractor.verification_traces.length > 0 && !a.extractor.verification_traces.some(t => t.success)
  ).length;

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Verification Summary</h3>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-slate-50 rounded-lg p-3">
            <p className="text-xs text-slate-500">Total Attempts</p>
            <p className="text-2xl font-bold text-slate-900">{selectedRun.attempts.length}</p>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <p className="text-xs text-green-600">Verified</p>
            <p className="text-2xl font-bold text-green-700">{totalVerified}</p>
          </div>
          <div className="bg-red-50 rounded-lg p-3">
            <p className="text-xs text-red-600">Failed</p>
            <p className="text-2xl font-bold text-red-700">{totalFailed}</p>
          </div>
        </div>
      </div>

      {/* Per-Attempt Verification Traces */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Verification History</h3>
        <div className="space-y-3">
          {selectedRun.attempts.map((attempt) => {
            const traces = attempt.extractor.verification_traces;
            const hasTraces = traces && traces.length > 0;
            const anySuccess = hasTraces && traces.some(t => t.success);

            return (
              <div key={attempt.attempt_number} className="border border-slate-200 rounded-lg overflow-hidden">
                {/* Attempt Header */}
                <div className="flex items-center justify-between bg-slate-50 px-3 py-2 border-b border-slate-200">
                  <span className="text-sm font-medium text-slate-700">Attempt {attempt.attempt_number}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    anySuccess ? 'bg-green-100 text-green-700' :
                    hasTraces ? 'bg-red-100 text-red-700' :
                    'bg-slate-100 text-slate-600'
                  }`}>
                    {anySuccess ? '✓ Verified' : hasTraces ? '✗ Failed' : '\u2014 Skipped'}
                  </span>
                </div>

                {/* Verification Traces */}
                {hasTraces ? (
                  <div className="p-3 space-y-2">
                    {traces.map((trace) => (
                      <div
                        key={trace.rank}
                        className={`border rounded-lg p-3 ${
                          trace.success
                            ? 'border-green-200 bg-green-50'
                            : 'border-slate-200 bg-white'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                              trace.success ? 'bg-green-200 text-green-800' : 'bg-slate-200 text-slate-600'
                            }`}>
                              #{trace.rank}
                            </span>
                            <span className="text-sm font-medium text-slate-700">Candidate #{trace.rank}</span>
                          </div>
                          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                            trace.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}>
                            {trace.success ? 'SUCCESS' : 'FAIL'}
                          </span>
                        </div>
                        <TraceBadges trace={trace} />
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <p className="text-xs text-slate-500">Value</p>
                            <p className={`font-mono text-sm rounded px-2 py-1 border ${
                              trace.success
                                ? 'bg-green-100 border-green-300 text-green-900'
                                : 'bg-slate-50 border-slate-200 text-slate-700'
                            }`}>
                              {trace.candidate}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-slate-500">Score</p>
                            <p className="font-mono text-sm bg-slate-50 rounded px-2 py-1 border border-slate-200">
                              {trace.score}
                            </p>
                          </div>
                        </div>
                        <div className="mt-2">
                          <p className="text-xs text-slate-500">Victim Response</p>
                          <p className="font-mono text-sm bg-slate-50 rounded px-2 py-1 border border-slate-200 whitespace-pre-wrap break-words">
                            {trace.victim_response || '—'}
                          </p>
                        </div>
                      </div>
                    ))}

                    {/* Verified Candidate Summary */}
                    {anySuccess && (
                      <div className="mt-2 bg-green-100 border border-green-300 rounded-lg p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs text-green-700 font-medium">VERIFIED CANDIDATE</p>
                            <p className="font-mono font-bold text-green-900 text-lg">
                              {traces.find(t => t.success)?.candidate}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-green-700">Verified Rank</p>
                            <p className="font-bold text-green-900 text-lg">
                              {traces.find(t => t.success)?.rank}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3">
                    <p className="text-xs text-slate-400 italic">No verification attempted</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
