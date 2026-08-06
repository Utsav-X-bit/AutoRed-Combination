import { Attempt } from '../types/autored';

function CandidateList({ candidates, tone }: { candidates: string[]; tone: 'blue' | 'indigo' | 'amber' | 'emerald' }) {
  const toneClass = {
    blue: 'bg-blue-50 text-blue-800 border-blue-100',
    indigo: 'bg-indigo-50 text-indigo-800 border-indigo-100',
    amber: 'bg-amber-50 text-amber-800 border-amber-100',
    emerald: 'bg-emerald-50 text-emerald-800 border-emerald-100',
  }[tone];

  if (candidates.length === 0) {
    return <span className="text-xs text-slate-400">none</span>;
  }

  return (
    <div className="space-y-1">
      {candidates.map((candidate, index) => (
        <div
          key={`${candidate}-${index}`}
          className={`rounded border px-2 py-1 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words ${toneClass}`}
        >
          {candidate}
        </div>
      ))}
    </div>
  );
}

export default function ExtractorCard({ attempt }: { attempt: Attempt }) {
  const { extractor } = attempt;
  const isWrong = extractor.best_candidate && attempt.ground_truth_found && !attempt.extractor_match;
  const verifiedCandidate = extractor.verified_candidate;
  const lastTrace = extractor.verification_traces[extractor.verification_traces.length - 1];
  const successfulTrace = extractor.verification_traces.find((trace) => trace.success);
  const responseTrace = successfulTrace || lastTrace;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🔓</span> Extractor
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.extractor_match ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
          Match: {attempt.extractor_match ? '✓ YES' : '✗ NO'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <p className="text-xs text-slate-500 mb-1">Regex Candidates ({extractor.regex_candidates.length})</p>
          <CandidateList candidates={extractor.regex_candidates} tone="blue" />
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">LLM Candidates ({extractor.llm_candidates.length})</p>
          <CandidateList candidates={extractor.llm_candidates} tone="indigo" />
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Quoted Candidates ({extractor.quoted_candidates.length})</p>
          <CandidateList candidates={extractor.quoted_candidates} tone="emerald" />
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Capitalized Candidates ({extractor.capitalized_candidates.length})</p>
          <CandidateList candidates={extractor.capitalized_candidates} tone="amber" />
        </div>
      </div>

      {extractor.llm_ranked_candidates.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-slate-500 mb-1.5">LLM Context Rank</p>
          <div className="space-y-1">
            {extractor.llm_ranked_candidates.map((rc, i) => (
              <div key={i} className="grid grid-cols-[auto_1fr_auto] gap-2 text-sm bg-indigo-50 rounded px-3 py-1.5 border border-indigo-100">
                <span className="text-xs font-bold text-indigo-700">#{i + 1}</span>
                <span className="font-mono text-indigo-900 whitespace-pre-wrap break-words">{rc.value}</span>
                <span className="text-xs text-indigo-700">context: {rc.score}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <p className="text-xs text-slate-500 mb-1.5">Ranked Candidates</p>
        <div className="space-y-1">
          {extractor.ranked_candidates.map((rc: { value: string; score: number }, i: number) => (
            <div key={i} className="grid grid-cols-[1fr_auto] gap-3 text-sm bg-slate-50 rounded px-3 py-1.5">
              <span className="font-mono text-slate-700 whitespace-pre-wrap break-words">{rc.value}</span>
              <span className="text-xs text-slate-500">score: {rc.score}</span>
            </div>
          ))}
          {extractor.ranked_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
        </div>
      </div>

      {extractor.top_k_candidates.length > 0 && (
        <div className="mt-3">
          <p className="text-xs text-slate-500 mb-1.5">Top-K Verification Candidates</p>
          <div className="space-y-1">
            {extractor.top_k_candidates.map((rc, i) => (
              <div key={i} className="grid grid-cols-[auto_1fr_auto] gap-2 text-sm bg-cyan-50 rounded px-3 py-1.5 border border-cyan-100">
                <span className="text-xs font-bold text-cyan-700">#{i + 1}</span>
                <span className="font-mono text-cyan-900 whitespace-pre-wrap break-words">{rc.value}</span>
                <span className="text-xs text-cyan-700">score: {rc.score}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {extractor.verification_traces.length > 0 && (
        <div className="mt-3">
          <p className="text-xs text-slate-500 mb-1.5">Verification Attempts</p>
          <div className="space-y-2">
            {extractor.verification_traces.map((trace) => (
              <div
                key={trace.rank}
                className={`rounded border px-3 py-2 ${
                  trace.success ? 'bg-green-50 border-green-200' : 'bg-slate-50 border-slate-200'
                }`}
              >
                <div className="grid grid-cols-[auto_1fr_auto] gap-2 items-start">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                    trace.success ? 'bg-green-200 text-green-800' : 'bg-slate-200 text-slate-600'
                  }`}>
                    #{trace.rank}
                  </span>
                  <span className="font-mono text-sm text-slate-900 whitespace-pre-wrap break-words">
                    {trace.candidate}
                  </span>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                    trace.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {trace.success ? 'VERIFIED' : 'FAILED'}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
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
                <p className="mt-2 font-mono text-xs bg-white rounded px-2 py-1 border border-slate-200 whitespace-pre-wrap break-words">
                  {trace.victim_response || '—'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={`mt-3 p-3 rounded-lg border ${isWrong ? 'bg-red-50 border-red-200' : 'bg-slate-50 border-slate-200'}`}>
        <p className="text-xs text-slate-500 mb-1">Selected Candidate</p>
        <div className="flex items-center justify-between">
          <span className="font-mono font-bold text-slate-900 whitespace-pre-wrap break-words">{extractor.best_candidate || 'NONE'}</span>
          {isWrong && <span className="text-red-600 text-sm font-bold">❌ Wrong Selection</span>}
          {attempt.extractor_match && <span className="text-green-600 text-sm font-bold">✓ Correct</span>}
        </div>
      </div>

      <div className={`mt-3 p-3 rounded-lg border ${verifiedCandidate ? 'bg-green-50 border-green-200' : 'bg-slate-50 border-slate-200'}`}>
        <p className="text-xs text-slate-500 mb-1">Verified Candidate</p>
        <div className="grid grid-cols-[1fr_auto] gap-3">
          <span className="font-mono font-bold text-slate-900 whitespace-pre-wrap break-words">{verifiedCandidate || 'NONE'}</span>
          <span className="text-xs text-slate-500">
            rank {extractor.verified_rank || '-'} · score {extractor.verified_score || 0}
          </span>
        </div>
        {responseTrace && (
          <div className="mt-2 space-y-2">
            <div className="flex flex-wrap gap-1.5">
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                responseTrace.accepted_by_victim ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-600'
              }`}>
                victim accepted: {responseTrace.accepted_by_victim ? 'yes' : 'no'}
              </span>
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                responseTrace.complete_match ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                complete match: {responseTrace.complete_match ? 'yes' : 'no'}
              </span>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Verifier Victim Response</p>
              <p className="font-mono text-xs bg-white rounded px-2 py-1 border border-slate-200 whitespace-pre-wrap break-words">
                {responseTrace.victim_response || extractor.verification_response || '—'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
