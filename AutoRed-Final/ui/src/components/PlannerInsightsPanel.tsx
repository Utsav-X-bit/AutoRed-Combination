import type { AutoRedRun } from '../types/autored';
import { attemptTimeline, buildPlannerState, kbRagSummary, parsePlanText, strategyRows } from '../utils/planner';

interface Props {
  run: AutoRedRun;
  selectedAttemptIndex: number;
}

const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;
const formatNumber = (value: number) => (Number.isFinite(value) ? value.toFixed(2) : 'n/a');

export default function PlannerInsightsPanel({ run, selectedAttemptIndex }: Props) {
  const attempt = run.attempts[selectedAttemptIndex];
  const parsedPlan = parsePlanText(attempt?.generator.plan_raw, attempt?.generator);
  const state = buildPlannerState(run, selectedAttemptIndex);
  const timeline = attemptTimeline(run);
  const rows = strategyRows(run.strategy_stats);
  const kbRag = kbRagSummary(run);

  return (
    <div className="space-y-4 rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-display text-base font-semibold text-stone-900 dark:text-stone-100">Planner analysis</h3>
          <p className="max-w-2xl text-xs text-stone-500 dark:text-stone-400">
            Planner contract output, best-known plan, state progression, and KB/RAG signals.
          </p>
        </div>
        <div className="text-right text-xs text-stone-500 dark:text-stone-400">
          <div>Attempt {state.attempt_number}</div>
          <div>{state.success_so_far} success(es) so far</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Current planner output</h4>
            <span className="text-xs text-stone-400 dark:text-stone-500">{attempt?.generator.plan_raw ? 'new pipeline' : 'legacy run'}</span>
          </div>
          {parsedPlan ? (
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <Field label="Strategy" value={parsedPlan.strategy} />
                <Field label="Style" value={parsedPlan.style} />
                <Field label="Access type" value={parsedPlan.expected_access_type} />
                <Field label="Retry" value={parsedPlan.retry_policy} />
                <Field label="Confidence" value={parsedPlan.confidence >= 0 ? formatPct(parsedPlan.confidence) : 'n/a'} />
                <Field label="Failure reason" value={parsedPlan.failure_reason} />
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Primitives</p>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {parsedPlan.primitives.length ? parsedPlan.primitives.map((primitive) => (
                    <span key={primitive} className="rounded-md border border-stone-200 bg-white px-2 py-1 text-xs text-stone-700 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-300">
                      {primitive}
                    </span>
                  )) : <span className="text-xs text-stone-400 dark:text-stone-500">No explicit primitive list.</span>}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-stone-500 dark:text-stone-400">Older runs only store strategy and generated attack.</p>
          )}
          {attempt?.generator.plan_raw && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs font-medium text-teal-700 hover:text-teal-800 dark:text-teal-400 dark:hover:text-teal-300">Show raw plan XML</summary>
              <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-white p-3 text-xs dark:border-stone-800 dark:bg-stone-950 dark:text-stone-300">
                {attempt.generator.plan_raw}
              </pre>
            </details>
          )}
        </div>

        <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
          <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">State</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <Field label="Current strategy" value={state.current_strategy} />
            <Field label="Previous strategy" value={state.previous_strategy} />
            <Field label="Previous outcome" value={state.previous_outcome} />
            <Field label="Attempts" value={`${state.attempts_so_far}`} />
            <Field label="Leaked so far" value={state.leak_seen ? 'yes' : 'no'} />
            <Field label="Verified so far" value={state.verified_seen ? 'yes' : 'no'} />
          </div>
          <p className="mt-3 text-xs text-stone-500 dark:text-stone-500">
            State is derived from run history when no explicit planner snapshot exists.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Best plan</h4>
            <span className="text-xs text-stone-400 dark:text-stone-500">{run.best_attack ? 'available' : 'missing'}</span>
          </div>
          {run.best_attack ? (
            <div className="space-y-2 text-sm">
              <Field label="Best strategy" value={run.best_attack.strategy} />
              <Field label="Best score" value={formatNumber(run.best_attack.score)} />
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Best attack prompt</p>
                <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-white p-3 text-xs dark:border-stone-800 dark:bg-stone-950 dark:text-stone-300">
                  {run.best_attack.prompt}
                </pre>
              </div>
            </div>
          ) : (
            <p className="text-sm text-stone-500 dark:text-stone-400">No best plan was recorded for this run.</p>
          )}
        </div>

        <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
          <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">KB / RAG usage</h4>
          <div className="space-y-2 text-sm">
            {kbRag.map((item) => (
              <div key={item.name} className="rounded-lg border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-950">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{item.name}</span>
                  <span className="text-xs text-stone-400 dark:text-stone-500">{item.summary}</span>
                </div>
                <p className="mt-2 text-sm text-stone-700 dark:text-stone-300">{item.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
        <div className="mb-2 flex items-center justify-between gap-3">
          <h4 className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Strategy performance</h4>
          <span className="text-xs text-stone-400 dark:text-stone-500">{rows.length} strategies</span>
        </div>
        <div className="overflow-x-auto rounded-lg border border-stone-200 dark:border-stone-800">
          <table className="w-full text-sm">
            <thead className="bg-stone-100 text-left text-xs font-semibold uppercase tracking-wider text-stone-600 dark:bg-stone-800 dark:text-stone-400">
              <tr>
                <th className="px-3 py-2">Strategy</th>
                <th className="px-3 py-2 text-right">Successes</th>
                <th className="px-3 py-2 text-right">Partial</th>
                <th className="px-3 py-2 text-right">Failures</th>
                <th className="px-3 py-2 text-right">Rate</th>
                <th className="px-3 py-2 text-right">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100 dark:divide-stone-800 bg-white dark:bg-stone-900">
              {rows.map((row) => (
                <tr key={row.strategy} className="hover:bg-stone-50 dark:hover:bg-stone-800/50">
                  <td className="px-3 py-2 font-mono text-xs text-stone-700 dark:text-stone-300">{row.strategy}</td>
                  <td className="px-3 py-2 text-right text-stone-900 dark:text-stone-100">{row.successes}</td>
                  <td className="px-3 py-2 text-right text-stone-900 dark:text-stone-100">{row.partial_leaks}</td>
                  <td className="px-3 py-2 text-right text-stone-900 dark:text-stone-100">{row.failures}</td>
                  <td className="px-3 py-2 text-right font-medium text-stone-900 dark:text-stone-100">{formatPct(row.successRate)}</td>
                  <td className="px-3 py-2 text-right text-stone-900 dark:text-stone-100">{formatNumber(row.total_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-lg border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/50">
        <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Timeline</h4>
        <div className="max-h-72 space-y-2 overflow-auto pr-1">
          {timeline.map((item) => (
            <div
              key={item.attempt_number}
              className={`rounded-lg border bg-white p-3 dark:bg-stone-950 ${item.attempt_number === state.attempt_number ? 'border-teal-300 ring-1 ring-teal-200 dark:border-teal-700 dark:ring-teal-900/50' : 'border-stone-200 dark:border-stone-800'}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-stone-900 dark:text-stone-100">Attempt {item.attempt_number}</p>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{item.strategy} · {item.time_ms} ms</p>
                </div>
                <span className={`text-xs font-semibold ${item.success ? 'text-emerald-600 dark:text-emerald-400' : 'text-stone-500 dark:text-stone-500'}`}>
                  {item.success ? 'success' : 'open'}
                </span>
              </div>
              <p className="mt-2 text-xs text-stone-600 dark:text-stone-400">
                State: {item.state.previous_outcome}. {item.state.success_so_far} success signal(s) in {item.state.attempts_so_far} attempt(s).
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white p-2 dark:border-stone-800 dark:bg-stone-900">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{label}</p>
      <p className="break-words text-sm font-medium text-stone-900 dark:text-stone-100">{value}</p>
    </div>
  );
}
