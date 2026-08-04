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
    if (!ground_truth_found) return 'No ground truth leak in victim output';
    if (extractor.ranked_candidates.some((c) => c.value.toLowerCase() === accessCode.toLowerCase())) {
      return 'Access code was in candidates but not selected as best';
    }
    if (extractor.regex_candidates.length === 0 && extractor.llm_candidates.length === 0) {
      return 'No candidates found by any extraction layer';
    }
    return 'Access code not found in any candidate list';
  })();

  return (
    <div className="space-y-4 pb-8">
      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Ground Truth Analysis</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-stone-200 p-3 dark:border-stone-800">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Ground Truth</p>
            <p className="font-mono text-lg font-bold text-amber-700 dark:text-amber-400">{accessCode}</p>
          </div>
          <div className="rounded-lg border border-stone-200 p-3 dark:border-stone-800">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Extractor Selected</p>
            <p className="font-mono text-lg font-bold text-stone-900 dark:text-stone-100">
              {extractor.best_candidate || 'NONE'}
              {extractor_match ? <span className="ml-2 text-emerald-600 dark:text-emerald-400">✓</span> : null}
              {!extractor_match && extractor.best_candidate ? <span className="ml-2 text-rose-600 dark:text-rose-400">✗</span> : null}
            </p>
          </div>
        </div>
      </section>

      {failureReason && (
        <section className="rounded-xl border border-rose-200 bg-rose-50 p-4 dark:border-rose-900/50 dark:bg-rose-950/20">
          <h4 className="mb-1 text-sm font-semibold text-rose-900 dark:text-rose-400">Extraction Failure</h4>
          <p className="text-sm text-rose-700 dark:text-rose-300">{failureReason}</p>
        </section>
      )}

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Extraction Trace</h3>
        <div className="space-y-3">
          <Layer
            title={`Layer 1: Regex Patterns (${extractor.regex_candidates.length} found)`}
            trigger={extractor.regex_candidates.some((c) => c.toLowerCase() === accessCode.toLowerCase())}
          >
            <CandidateList values={extractor.regex_candidates} accessCode={accessCode} />
          </Layer>

          <Layer title={`Layer 2: Quoted Strings (${extractor.quoted_candidates.length} found)`} trigger={false}>
            <CandidateList values={extractor.quoted_candidates} accessCode={accessCode} />
          </Layer>

          <Layer title={`Layer 3: Capitalized Words (${extractor.capitalized_candidates.length} found)`} trigger={false}>
            <CandidateList values={extractor.capitalized_candidates} accessCode={accessCode} />
          </Layer>

          <Layer
            title={`Layer 4: LLM Candidates (${extractor.llm_candidates.length} found)`}
            trigger={extractor.llm_candidates.some((c) => c.toLowerCase() === accessCode.toLowerCase())}
          >
            <CandidateList values={extractor.llm_candidates} accessCode={accessCode} kind="llm" />
          </Layer>

          {extractor.llm_ranked_candidates.length > 0 && (
            <div>
              <p className="mb-1 text-xs font-medium text-stone-700 dark:text-stone-300">Layer 4b: LLM Context Rank</p>
              <div className="space-y-1">
                {extractor.llm_ranked_candidates.map((rc, i) => {
                  const isGT = rc.value.toLowerCase() === accessCode.toLowerCase();
                  return (
                    <div key={i} className={`flex items-center justify-between rounded-lg border px-3 py-1.5 text-sm ${isGT ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20' : 'border-indigo-100 bg-indigo-50 dark:border-indigo-900/30 dark:bg-indigo-950/20'}`}>
                      <div className="flex items-center gap-2">
                        <span className="w-4 text-xs text-stone-500 dark:text-stone-400">{i + 1}</span>
                        <span className="whitespace-pre-wrap break-words font-mono text-stone-700 dark:text-stone-300">{rc.value}</span>
                        {isGT && <span className="text-xs text-emerald-600 dark:text-emerald-400">GT</span>}
                      </div>
                      <span className="text-xs text-indigo-700 dark:text-indigo-400">context {rc.score}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div>
            <p className="mb-1 text-xs font-medium text-stone-700 dark:text-stone-300">Layer 5: Ranked Candidates</p>
            <div className="space-y-1">
              {extractor.ranked_candidates.map((rc, i) => {
                const isGT = rc.value.toLowerCase() === accessCode.toLowerCase();
                const isSelected = rc.value === extractor.best_candidate;
                return (
                  <div key={i} className={`flex items-center justify-between rounded-lg border px-3 py-1.5 text-sm ${isSelected ? 'border-indigo-200 bg-indigo-50 dark:border-indigo-900/50 dark:bg-indigo-950/20' : 'border-stone-100 bg-stone-50 dark:border-stone-800 dark:bg-stone-900/50'}`}>
                    <div className="flex items-center gap-2">
                      <span className="w-4 text-xs text-stone-500 dark:text-stone-400">{i + 1}</span>
                      <span className="whitespace-pre-wrap break-words font-mono text-stone-900 dark:text-stone-100">{rc.value}</span>
                      {isGT && <span className="text-xs text-emerald-600 dark:text-emerald-400">GT</span>}
                      {isSelected && <span className="text-xs text-indigo-600 dark:text-indigo-400">selected</span>}
                    </div>
                    <span className="text-xs text-stone-500 dark:text-stone-400">{rc.score}</span>
                  </div>
                );
              })}
              {extractor.ranked_candidates.length === 0 && <span className="text-xs text-stone-400 dark:text-stone-500">none</span>}
            </div>
          </div>

          <div>
            <p className="mb-1 text-xs font-medium text-stone-700 dark:text-stone-300">Layer 6: Final Selection</p>
            <div className={`rounded-lg border p-3 ${extractor_match ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20' : 'border-rose-200 bg-rose-50 dark:border-rose-900/50 dark:bg-rose-950/20'}`}>
              <div className="flex items-center justify-between">
                <span className="whitespace-pre-wrap break-words font-mono font-bold text-stone-900 dark:text-stone-100">{extractor.best_candidate || 'NONE'}</span>
                {extractor_match ? <span className="font-bold text-emerald-600 dark:text-emerald-400">Correct</span> : <span className="font-bold text-rose-600 dark:text-rose-400">Wrong</span>}
              </div>
            </div>
          </div>

          {extractor.verification_traces && extractor.verification_traces.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-stone-700 dark:text-stone-300">Layer 7: Verification Loop</p>
              <div className="space-y-2">
                {extractor.verification_traces.map((trace: VerificationTraceItem) => (
                  <VerificationTraceItemCard key={trace.rank} trace={trace} />
                ))}

                {extractor.verified_candidate && (
                  <div className="rounded-lg border border-emerald-300 bg-emerald-100 p-3 dark:border-emerald-800 dark:bg-emerald-950/40">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs font-medium text-emerald-800 dark:text-emerald-400">VERIFIED CANDIDATE</p>
                        <p className="font-mono text-lg font-bold text-emerald-900 dark:text-emerald-300">
                          <CandidateText value={extractor.verified_candidate} />
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-emerald-800 dark:text-emerald-400">Verified Rank</p>
                        <p className="text-lg font-bold text-emerald-900 dark:text-emerald-300">{extractor.verified_rank}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function Layer({ title, trigger, children }: { title: string; trigger: boolean; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <p className="text-xs font-medium text-stone-700 dark:text-stone-300">{title}</p>
        <span className={`rounded-full px-2 py-0.5 text-xs ${trigger ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
          {trigger ? 'contains GT' : 'checked'}
        </span>
      </div>
      {children}
    </div>
  );
}

function CandidateList({ values, accessCode, kind }: { values: string[]; accessCode: string; kind?: 'llm' }) {
  return (
    <div className="flex flex-wrap gap-1">
      {values.map((c, i) => (
        <span
          key={i}
          className={`rounded px-2 py-0.5 font-mono text-xs ${
            c.toLowerCase() === accessCode.toLowerCase()
              ? 'border border-emerald-300 bg-emerald-100 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
              : kind === 'llm'
              ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/30 dark:text-indigo-400'
              : 'bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300'
          }`}
        >
          <CandidateText value={c} />
        </span>
      ))}
      {values.length === 0 && <span className="text-xs text-stone-400 dark:text-stone-500">none</span>}
    </div>
  );
}

function VerificationTraceItemCard({ trace }: { trace: VerificationTraceItem }) {
  return (
    <div className={`rounded-lg border p-3 ${trace.success ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20' : 'border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900'}`}>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`rounded px-1.5 py-0.5 text-xs font-bold ${trace.success ? 'bg-emerald-200 text-emerald-800 dark:bg-emerald-900/70 dark:text-emerald-300' : 'bg-stone-200 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
            #{trace.rank}
          </span>
          <span className="text-sm font-medium text-stone-700 dark:text-stone-300">Candidate #{trace.rank}</span>
        </div>
        <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${trace.success ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400' : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'}`}>
          {trace.success ? 'SUCCESS' : 'FAIL'}
        </span>
      </div>
      <div className="mb-2 flex flex-wrap gap-1.5">
        <Badge color={trace.accepted_by_victim ? 'amber' : 'muted'}>victim accepted: {trace.accepted_by_victim ? 'yes' : 'no'}</Badge>
        <Badge color={trace.complete_match ? 'emerald' : 'rose'}>complete match: {trace.complete_match ? 'yes' : 'no'}</Badge>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <p className="text-xs text-stone-500 dark:text-stone-400">Value</p>
          <p className={`rounded border px-2 py-1 font-mono text-sm ${trace.success ? 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300' : 'border-stone-200 bg-stone-50 text-stone-700 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300'}`}>
            <CandidateText value={trace.candidate} />
          </p>
        </div>
        <div>
          <p className="text-xs text-stone-500 dark:text-stone-400">Score</p>
          <p className="rounded border border-stone-200 bg-stone-50 px-2 py-1 font-mono text-sm dark:border-stone-800 dark:bg-stone-900">{trace.score}</p>
        </div>
      </div>
      <div className="mt-2">
        <p className="text-xs text-stone-500 dark:text-stone-400">Victim Response</p>
        <p className="whitespace-pre-wrap break-words rounded border border-stone-200 bg-stone-50 px-2 py-1 font-mono text-sm dark:border-stone-800 dark:bg-stone-900">{trace.victim_response || '—'}</p>
      </div>
    </div>
  );
}

function Badge({ children, color }: { children: React.ReactNode; color: 'emerald' | 'rose' | 'amber' | 'muted' }) {
  const map = {
    emerald: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400',
    rose: 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400',
    amber: 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-400',
    muted: 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400',
  };
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${map[color]}`}>{children}</span>;
}
