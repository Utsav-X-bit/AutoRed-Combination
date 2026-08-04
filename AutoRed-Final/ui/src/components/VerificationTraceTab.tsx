import { useRunStore } from '../store/runStore';
import type { VerificationTraceItem } from '../types/autored';

function TraceBadges({ trace }: { trace: VerificationTraceItem }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      <Badge color={trace.accepted_by_victim ? 'amber' : 'muted'}>victim accepted: {trace.accepted_by_victim ? 'yes' : 'no'}</Badge>
      <Badge color={trace.complete_match ? 'emerald' : 'rose'}>complete match: {trace.complete_match ? 'yes' : 'no'}</Badge>
    </div>
  );
}

export default function VerificationTraceTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const totalVerified = selectedRun.attempts.filter(
    (a) => a.extractor.verification_traces?.some((t) => t.success)
  ).length;

  const totalFailed = selectedRun.attempts.filter(
    (a) =>
      a.extractor.verification_traces &&
      a.extractor.verification_traces.length > 0 &&
      !a.extractor.verification_traces.some((t) => t.success)
  ).length;

  return (
    <div className="space-y-4 pb-8">
      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Verification Summary</h3>
        <div className="grid grid-cols-3 gap-3 text-center">
          <StatCard label="Total Attempts" value={String(selectedRun.attempts.length)} />
          <StatCard label="Verified" value={String(totalVerified)} accent="emerald" />
          <StatCard label="Failed" value={String(totalFailed)} accent="rose" />
        </div>
      </section>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Verification History</h3>
        <div className="space-y-3">
          {selectedRun.attempts.map((attempt) => {
            const traces = attempt.extractor.verification_traces;
            const hasTraces = traces && traces.length > 0;
            const anySuccess = hasTraces && traces.some((t) => t.success);

            return (
              <div key={attempt.attempt_number} className="overflow-hidden rounded-lg border border-stone-200 dark:border-stone-800">
                <div className="flex items-center justify-between border-b border-stone-200 bg-stone-50 px-3 py-2 dark:border-stone-800 dark:bg-stone-800/50">
                  <span className="text-sm font-medium text-stone-900 dark:text-stone-100">Attempt {attempt.attempt_number}</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    anySuccess
                      ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
                      : hasTraces
                      ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'
                      : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'
                  }`}>
                    {anySuccess ? '✓ Verified' : hasTraces ? '✗ Failed' : '— Skipped'}
                  </span>
                </div>

                {hasTraces ? (
                  <div className="space-y-2 p-3">
                    {traces.map((trace) => (
                      <div
                        key={trace.rank}
                        className={`rounded-lg border p-3 ${
                          trace.success
                            ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20'
                            : 'border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900'
                        }`}
                      >
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
                        <TraceBadges trace={trace} />
                        <div className="mt-2 grid grid-cols-2 gap-2">
                          <div>
                            <p className="text-xs text-stone-500 dark:text-stone-400">Value</p>
                            <p className={`rounded border px-2 py-1 font-mono text-sm ${trace.success ? 'border-emerald-300 bg-emerald-100 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300' : 'border-stone-200 bg-stone-50 text-stone-700 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300'}`}>
                              {trace.candidate}
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
                    ))}

                    {anySuccess && (
                      <div className="rounded-lg border border-emerald-300 bg-emerald-100 p-3 dark:border-emerald-800 dark:bg-emerald-950/40">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs font-medium text-emerald-800 dark:text-emerald-400">VERIFIED CANDIDATE</p>
                            <p className="font-mono text-lg font-bold text-emerald-900 dark:text-emerald-300">{traces.find((t) => t.success)?.candidate}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs text-emerald-800 dark:text-emerald-400">Verified Rank</p>
                            <p className="text-lg font-bold text-emerald-900 dark:text-emerald-300">{traces.find((t) => t.success)?.rank}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3">
                    <p className="text-xs italic text-stone-400 dark:text-stone-500">No verification attempted</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent?: 'emerald' | 'rose' }) {
  const valueClass = accent === 'emerald'
    ? 'text-emerald-700 dark:text-emerald-400'
    : accent === 'rose'
    ? 'text-rose-700 dark:text-rose-400'
    : 'text-stone-900 dark:text-stone-100';
  return (
    <div className="rounded-lg border border-stone-100 p-3 dark:border-stone-800">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className={`font-display text-2xl font-bold ${valueClass}`}>{value}</p>
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
