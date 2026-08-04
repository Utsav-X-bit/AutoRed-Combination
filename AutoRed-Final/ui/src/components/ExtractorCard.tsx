import { Attempt } from '../types/autored';

function CandidateList({ candidates, tone }: { candidates: string[]; tone: 'blue' | 'indigo' | 'amber' | 'emerald' }) {
  const toneClass = {
    blue: 'bg-blue-50 text-blue-800 border-blue-100 dark:bg-blue-950/30 dark:text-blue-300 dark:border-blue-900/40',
    indigo: 'bg-indigo-50 text-indigo-800 border-indigo-100 dark:bg-indigo-950/30 dark:text-indigo-300 dark:border-indigo-900/40',
    amber: 'bg-amber-50 text-amber-800 border-amber-100 dark:bg-amber-950/30 dark:text-amber-300 dark:border-amber-900/40',
    emerald: 'bg-emerald-50 text-emerald-800 border-emerald-100 dark:bg-emerald-950/30 dark:text-emerald-300 dark:border-emerald-900/40',
  }[tone];

  if (candidates.length === 0) {
    return <span className="text-xs text-stone-400 dark:text-stone-500">none</span>;
  }

  return (
    <div className="space-y-1">
      {candidates.map((candidate, index) => (
        <div
          key={`${candidate}-${index}`}
          className={`break-words whitespace-pre-wrap rounded border px-2 py-1 font-mono text-xs leading-relaxed ${toneClass}`}
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
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">🔓</span>
          Extractor
        </h3>
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${attempt.extractor_match ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
          Match: {attempt.extractor_match ? 'YES' : 'NO'}
        </span>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-3">
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Regex ({extractor.regex_candidates.length})</p>
          <CandidateList candidates={extractor.regex_candidates} tone="blue" />
        </div>
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">LLM ({extractor.llm_candidates.length})</p>
          <CandidateList candidates={extractor.llm_candidates} tone="indigo" />
        </div>
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Quoted ({extractor.quoted_candidates.length})</p>
          <CandidateList candidates={extractor.quoted_candidates} tone="emerald" />
        </div>
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Capitalized ({extractor.capitalized_candidates.length})</p>
          <CandidateList candidates={extractor.capitalized_candidates} tone="amber" />
        </div>
      </div>

      {extractor.llm_ranked_candidates.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">LLM Context Rank</p>
          <div className="space-y-1">
            {extractor.llm_ranked_candidates.map((rc, i) => (
              <div key={i} className="grid grid-cols-[auto_1fr_auto] gap-2 rounded-md border border-indigo-100 bg-indigo-50 px-3 py-1.5 text-sm dark:border-indigo-900/40 dark:bg-indigo-950/30">
                <span className="text-xs font-bold text-indigo-700 dark:text-indigo-300">#{i + 1}</span>
                <span className="break-words whitespace-pre-wrap font-mono text-indigo-900 dark:text-indigo-200">{rc.value}</span>
                <span className="text-xs text-indigo-700 dark:text-indigo-300">ctx {rc.score}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Ranked Candidates</p>
        <div className="space-y-1">
          {extractor.ranked_candidates.map((rc, i) => (
            <div key={i} className="grid grid-cols-[1fr_auto] gap-3 rounded-md bg-stone-50 px-3 py-1.5 dark:bg-stone-800/50">
              <span className="break-words whitespace-pre-wrap font-mono text-sm text-stone-700 dark:text-stone-300">{rc.value}</span>
              <span className="text-xs text-stone-500 dark:text-stone-400">{rc.score}</span>
            </div>
          ))}
          {extractor.ranked_candidates.length === 0 && <span className="text-xs text-stone-400 dark:text-stone-500">none</span>}
        </div>
      </div>

      {extractor.top_k_candidates.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Top-K Verification Candidates</p>
          <div className="space-y-1">
            {extractor.top_k_candidates.map((rc, i) => (
              <div key={i} className="grid grid-cols-[auto_1fr_auto] gap-2 rounded-md border border-cyan-100 bg-cyan-50 px-3 py-1.5 text-sm dark:border-cyan-900/40 dark:bg-cyan-950/30">
                <span className="text-xs font-bold text-cyan-700 dark:text-cyan-300">#{i + 1}</span>
                <span className="break-words whitespace-pre-wrap font-mono text-cyan-900 dark:text-cyan-200">{rc.value}</span>
                <span className="text-xs text-cyan-700 dark:text-cyan-300">{rc.score}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {extractor.verification_traces.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Verification Attempts</p>
          <div className="space-y-2">
            {extractor.verification_traces.map((trace) => (
              <div
                key={trace.rank}
                className={`rounded-lg border px-3 py-2 ${trace.success ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/40 dark:bg-emerald-950/20' : 'border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-800/30'}`}
              >
                <div className="grid grid-cols-[auto_1fr_auto] items-start gap-2">
                  <span className={`rounded px-1.5 py-0.5 text-xs font-bold ${trace.success ? 'bg-emerald-200 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-200' : 'bg-stone-200 text-stone-600 dark:bg-stone-700 dark:text-stone-300'}`}>
                    #{trace.rank}
                  </span>
                  <span className="break-words whitespace-pre-wrap font-mono text-sm text-stone-900 dark:text-stone-100">{trace.candidate}</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${trace.success ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'}`}>
                    {trace.success ? 'VERIFIED' : 'FAILED'}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${trace.accepted_by_victim ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
                    victim accepted: {trace.accepted_by_victim ? 'yes' : 'no'}
                  </span>
                  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${trace.complete_match ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-400'}`}>
                    complete match: {trace.complete_match ? 'yes' : 'no'}
                  </span>
                </div>
                <p className="mt-2 break-words whitespace-pre-wrap rounded border border-stone-200 bg-white px-2 py-1 font-mono text-xs text-stone-700 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300">
                  {trace.victim_response || '—'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={`mt-3 rounded-lg border p-3 ${isWrong ? 'border-rose-200 bg-rose-50 dark:border-rose-900/40 dark:bg-rose-950/20' : 'border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-800/30'}`}>
        <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Selected Candidate</p>
        <div className="flex items-center justify-between">
          <span className="break-words whitespace-pre-wrap font-mono font-bold text-stone-900 dark:text-stone-100">{extractor.best_candidate || 'NONE'}</span>
          {isWrong && <span className="text-sm font-bold text-rose-600 dark:text-rose-400">❌ Wrong</span>}
          {attempt.extractor_match && <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400">✓ Correct</span>}
        </div>
      </div>

      <div className={`mt-3 rounded-lg border p-3 ${verifiedCandidate ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/40 dark:bg-emerald-950/20' : 'border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-800/30'}`}>
        <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Verified Candidate</p>
        <div className="grid grid-cols-[1fr_auto] gap-3">
          <span className="break-words whitespace-pre-wrap font-mono font-bold text-stone-900 dark:text-stone-100">{verifiedCandidate || 'NONE'}</span>
          <span className="text-xs text-stone-500 dark:text-stone-400">
            rank {extractor.verified_rank || '-'} · score {extractor.verified_score || 0}
          </span>
        </div>
        {responseTrace && (
          <div className="mt-2 space-y-2">
            <div className="flex flex-wrap gap-1.5">
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${responseTrace.accepted_by_victim ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
                victim accepted: {responseTrace.accepted_by_victim ? 'yes' : 'no'}
              </span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${responseTrace.complete_match ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-400'}`}>
                complete match: {responseTrace.complete_match ? 'yes' : 'no'}
              </span>
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Verifier Victim Response</p>
              <p className="mt-1 break-words whitespace-pre-wrap rounded border border-stone-200 bg-white px-2 py-1 font-mono text-xs dark:border-stone-800 dark:bg-stone-900">
                {responseTrace.victim_response || extractor.verification_response || '—'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
