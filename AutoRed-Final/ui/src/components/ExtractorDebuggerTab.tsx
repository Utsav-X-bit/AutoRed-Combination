import { useRunStore } from '../store/runStore';
import type { VerificationTraceItem } from '../types/autored';

function CandidateText({ value }: { value: string }) {
  return <span className="whitespace-pre-wrap break-words">{value}</span>;
}

export default function ExtractorDebuggerTab() {
  const { selectedRun, selectedAttemptIndex } = useRunStore();
  if (!selectedRun) return null;

  const attempt = selectedRun.attempts[selectedAttemptIndex];
  if (!attempt) return null;

  const { extractor, ground_truth_found, extractor_match } = attempt;
  const accessCode = selectedRun.scenario.access_code;

  const failureReason = (() => {
    if (extractor_match) return null;
    if (!ground_truth_found) return "No ground truth leak in victim output";
    if (extractor.ranked_candidates.some((c: { value: string }) => c.value.toLowerCase() === accessCode.toLowerCase())) {
      return "Access code was in candidates but not selected as best";
    }
    if (extractor.regex_candidates.length === 0 && extractor.llm_candidates.length === 0) {
      return "No candidates found by any extraction layer";
    }
    return "Access code not found in any candidate list";
  })();

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Ground Truth Analysis</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-50 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Ground Truth</p>
            <p className="font-mono font-bold text-lg text-amber-600">{accessCode}</p>
          </div>
          <div className="bg-slate-50 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">Extractor Selected</p>
            <p className="font-mono font-bold text-lg">
              {extractor.best_candidate || 'NONE'}
              {extractor_match ? <span className="text-green-600 ml-2">✓</span> : null}
              {!extractor_match && extractor.best_candidate ? <span className="text-red-600 ml-2">✗</span> : null}
            </p>
          </div>
        </div>
      </div>

      {failureReason && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <h4 className="text-sm font-bold text-red-900 mb-1">❌ Extraction Failure</h4>
          <p className="text-sm text-red-700">{failureReason}</p>
        </div>
      )}

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">6-Layer Extraction Trace</h3>
        <div className="space-y-3">
          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium text-slate-700">Layer 1: Regex Patterns</p>
              <span className={`text-xs px-2 py-0.5 rounded-full ${extractor.regex_candidates.some((c: string) => c.toLowerCase() === accessCode.toLowerCase()) ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                {extractor.regex_candidates.length} found
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {extractor.regex_candidates.map((c: string, i: number) => (
                  <span key={i} className={`px-2 py-0.5 text-xs rounded font-mono ${c.toLowerCase() === accessCode.toLowerCase() ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-blue-50 text-blue-700'}`}>
                    <CandidateText value={c} />
                </span>
              ))}
              {extractor.regex_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium text-slate-700">Layer 2: Quoted Strings</p>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">{extractor.quoted_candidates.length} found</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {extractor.quoted_candidates.map((c: string, i: number) => (
                  <span key={i} className={`px-2 py-0.5 text-xs rounded font-mono ${c.toLowerCase() === accessCode.toLowerCase() ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-blue-50 text-blue-700'}`}>
                    <CandidateText value={c} />
                </span>
              ))}
              {extractor.quoted_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium text-slate-700">Layer 3: Capitalized Words</p>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">{extractor.capitalized_candidates.length} found</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {extractor.capitalized_candidates.map((c: string, i: number) => (
                  <span key={i} className={`px-2 py-0.5 text-xs rounded font-mono ${c.toLowerCase() === accessCode.toLowerCase() ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-blue-50 text-blue-700'}`}>
                    <CandidateText value={c} />
                </span>
              ))}
              {extractor.capitalized_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium text-slate-700">Layer 4: LLM Candidates</p>
              <span className={`text-xs px-2 py-0.5 rounded-full ${extractor.llm_candidates.some((c: string) => c.toLowerCase() === accessCode.toLowerCase()) ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                {extractor.llm_candidates.length} found
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {extractor.llm_candidates.map((c: string, i: number) => (
                  <span key={i} className={`px-2 py-0.5 text-xs rounded font-mono ${c.toLowerCase() === accessCode.toLowerCase() ? 'bg-green-100 text-green-800 border border-green-300' : 'bg-indigo-50 text-indigo-700'}`}>
                    <CandidateText value={c} />
                </span>
              ))}
              {extractor.llm_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
            </div>
          </div>

          {extractor.llm_ranked_candidates.length > 0 && (
            <div>
              <p className="text-xs font-medium text-slate-700 mb-1">Layer 4b: LLM Context Rank</p>
              <div className="space-y-1">
                {extractor.llm_ranked_candidates.map((rc: { value: string; score: number }, i: number) => {
                  const isGT = rc.value.toLowerCase() === accessCode.toLowerCase();
                  return (
                    <div key={i} className={`flex items-center justify-between text-sm rounded px-3 py-1.5 ${isGT ? 'bg-green-50 border border-green-200' : 'bg-indigo-50 border border-indigo-100'}`}>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400 w-4">{i + 1}</span>
                        <span className="font-mono text-slate-700 whitespace-pre-wrap break-words">{rc.value}</span>
                        {isGT && <span className="text-xs text-green-600">GT</span>}
                      </div>
                      <span className="text-xs text-indigo-700">context {rc.score}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div>
            <p className="text-xs font-medium text-slate-700 mb-1">Layer 5: Ranked Candidates</p>
            <div className="space-y-1">
              {extractor.ranked_candidates.map((rc: { value: string; score: number }, i: number) => {
                const isGT = rc.value.toLowerCase() === accessCode.toLowerCase();
                const isSelected = rc.value === extractor.best_candidate;
                return (
                  <div key={i} className={`flex items-center justify-between text-sm rounded px-3 py-1.5 ${isSelected ? 'bg-purple-50 border border-purple-200' : 'bg-slate-50'}`}>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 w-4">{i + 1}</span>
                      <span className="font-mono text-slate-700 whitespace-pre-wrap break-words">{rc.value}</span>
                      {isGT && <span className="text-xs text-green-600">🎯 GT</span>}
                      {isSelected && <span className="text-xs text-purple-600">→ selected</span>}
                    </div>
                    <span className="text-xs text-slate-500">{rc.score}</span>
                  </div>
                );
              })}
              {extractor.ranked_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
            </div>
          </div>

          <div>
            <p className="text-xs font-medium text-slate-700 mb-1">Layer 6: Final Selection</p>
            <div className={`p-3 rounded-lg border ${extractor_match ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold whitespace-pre-wrap break-words">{extractor.best_candidate || 'NONE'}</span>
                {extractor_match ? <span className="text-green-600 font-bold">✓ Correct</span> : <span className="text-red-600 font-bold">✗ Wrong</span>}
              </div>
            </div>
          </div>

          {/* Layer 7: Verification Loop */}
          {extractor.verification_traces && extractor.verification_traces.length > 0 && (
            <div>
              <p className="text-xs font-medium text-slate-700 mb-1">Layer 7: Verification Loop</p>
              <div className="space-y-2">
                {extractor.verification_traces.map((trace: VerificationTraceItem) => (
                  <div
                    key={trace.rank}
                    className={`border rounded-lg p-3 ${
                      trace.success ? 'border-green-200 bg-green-50' : 'border-slate-200 bg-white'
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
                    <div className="flex flex-wrap gap-1.5 mb-2">
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
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <p className="text-xs text-slate-500">Value</p>
                        <p className={`font-mono text-sm rounded px-2 py-1 border ${
                          trace.success
                            ? 'bg-green-100 border-green-300 text-green-900'
                            : 'bg-slate-50 border-slate-200 text-slate-700'
                        }`}>
                          <CandidateText value={trace.candidate} />
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
                {extractor.verified_candidate && (
                  <div className="bg-green-100 border border-green-300 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-green-700 font-medium">VERIFIED CANDIDATE</p>
                        <p className="font-mono font-bold text-green-900 text-lg">
                          <CandidateText value={extractor.verified_candidate} />
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-green-700">Verified Rank</p>
                        <p className="font-bold text-green-900 text-lg">
                          {extractor.verified_rank}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
