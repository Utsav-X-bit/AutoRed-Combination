import { Attempt } from '../types/autored';
import { parsePlanText } from '../utils/planner';

export default function GeneratorCard({ attempt }: { attempt: Attempt }) {
  const parsedPlan = parsePlanText(attempt.generator.plan_raw, attempt.generator);

  return (
    <div className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-purple-100 text-purple-700 dark:bg-purple-950/50 dark:text-purple-300">🧠</span>
          Generator
        </h3>
        <span className="rounded-full bg-purple-100 px-2 py-0.5 text-xs font-semibold text-purple-700 dark:bg-purple-950/50 dark:text-purple-300">
          {attempt.generator.strategy}
        </span>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-3">
        <div className="rounded-lg bg-stone-50 p-2.5 dark:bg-stone-800/50">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Input Tokens</p>
          <p className="font-display text-lg font-semibold text-stone-900 dark:text-stone-100">{attempt.generator.input_tokens}</p>
        </div>
        <div className="rounded-lg bg-stone-50 p-2.5 dark:bg-stone-800/50">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Output Tokens</p>
          <p className="font-display text-lg font-semibold text-stone-900 dark:text-stone-100">{attempt.generator.output_tokens}</p>
        </div>
      </div>

      <div>
        <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Generated Attack</p>
        <div className="mt-1 rounded-lg border border-rose-200 bg-rose-50 p-3 dark:border-rose-900/50 dark:bg-rose-950/20">
          <p className="font-mono text-sm leading-relaxed text-rose-900 dark:text-rose-200">
            "{attempt.generator.generated_attack}"
          </p>
        </div>
      </div>

      <div className="mt-4 border-t border-stone-200 pt-4 dark:border-stone-800">
        <div className="mb-2 flex items-center justify-between gap-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Planner contract</p>
          <span className="text-xs text-stone-400 dark:text-stone-500">
            {attempt.generator.plan_raw ? 'new pipeline' : 'legacy run'}
          </span>
        </div>

        {parsedPlan ? (
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <Meta label="Strategy" value={parsedPlan.strategy} />
              <Meta label="Style" value={parsedPlan.style} />
              <Meta label="Access type" value={parsedPlan.expected_access_type} />
              <Meta label="Retry" value={parsedPlan.retry_policy} />
              <Meta label="Confidence" value={parsedPlan.confidence >= 0 ? parsedPlan.confidence.toFixed(2) : 'n/a'} />
              <Meta label="Failure reason" value={parsedPlan.failure_reason} />
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Primitives</p>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {parsedPlan.primitives.length ? parsedPlan.primitives.map((primitive) => (
                  <span key={primitive} className="rounded-md border border-stone-200 bg-stone-50 px-2 py-1 text-xs text-stone-700 dark:border-stone-800 dark:bg-stone-800 dark:text-stone-300">
                    {primitive}
                  </span>
                )) : <span className="text-xs text-stone-400 dark:text-stone-500">No explicit primitives in this run.</span>}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-stone-500 dark:text-stone-400">
            This run does not store the planner XML. The attack was generated from the legacy prompt path.
          </p>
        )}

        {attempt.generator.plan_raw && (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs font-medium text-teal-700 hover:text-teal-800 dark:text-teal-400 dark:hover:text-teal-300">
              Show raw planner XML
            </summary>
            <pre className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-stone-50 p-3 text-xs text-stone-700 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-300">
              {attempt.generator.plan_raw}
            </pre>
          </details>
        )}
      </div>

      {attempt.generator.duplicate_attack && (
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/20 dark:text-amber-400">
          ⚠️ Duplicate attack detected
        </div>
      )}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-stone-50 p-2 dark:border-stone-800 dark:bg-stone-800/50">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className="break-words text-sm font-medium text-stone-900 dark:text-stone-100">{value}</p>
    </div>
  );
}
