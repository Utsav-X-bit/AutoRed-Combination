import { Attempt } from '../types/autored';

export default function VerifierCard({ attempt }: { attempt: Attempt }) {
  const { verification } = attempt;

  return (
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-blue-100 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300">🔍</span>
          Verifier
        </h3>
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${verification.success ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400' : 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-400'}`}>
          {verification.success ? '✓ Verified' : '✗ Not Verified'}
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Candidate Sent</p>
          <p className="mt-1 rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 font-mono text-sm text-stone-900 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-100">
            {verification.candidate_sent || '—'}
          </p>
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Victim Response</p>
          <p className="mt-1 max-h-48 overflow-y-auto rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 font-mono text-sm text-stone-900 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-100">
            {verification.victim_response || '—'}
          </p>
        </div>
      </div>
    </div>
  );
}
