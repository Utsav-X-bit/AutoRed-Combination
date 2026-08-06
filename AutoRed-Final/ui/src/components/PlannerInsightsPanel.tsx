import type { AutoRedRun } from '../types/autored';
import { attemptTimeline, buildPlannerState, kbRagSummary, parsePlanText, strategyRows } from '../utils/planner';

interface Props {
  run: AutoRedRun;
  selectedAttemptIndex: number;
}

const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;

const formatNumber = (value: number) => Number.isFinite(value) ? value.toFixed(2) : 'n/a';

export default function PlannerInsightsPanel({ run, selectedAttemptIndex }: Props) {
  const attempt = run.attempts[selectedAttemptIndex];
  const parsedPlan = parsePlanText(attempt?.generator.plan_raw);
  const state = buildPlannerState(run, selectedAttemptIndex);
  const timeline = attemptTimeline(run);
  const rows = strategyRows(run.strategy_stats);
  const kbRag = kbRagSummary(run);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Planner analysis</h3>
          <p className="text-xs text-slate-500">
            Shows the contract output, the best-known plan, state progression, and the KB/RAG signals used for planning.
          </p>
        </div>
        <div className="text-right text-xs text-slate-500">
          <div>Attempt {state.attempt_number}</div>
          <div>{state.success_so_far} success(es) so far</div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
          <div className="flex items-center justify-between gap-3 mb-2">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Current planner output</h4>
            <span className="text-xs text-slate-500">
              {attempt?.generator.plan_raw ? 'new pipeline' : 'legacy run'}
            </span>
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
                <p className="text-xs text-slate-500 mb-1">Primitives</p>
                <div className="flex flex-wrap gap-1.5">
                  {parsedPlan.primitives.length ? parsedPlan.primitives.map((primitive) => (
                    <span key={primitive} className="px-2 py-1 rounded-md bg-white border border-slate-200 text-xs text-slate-700">
                      {primitive}
                    </span>
                  )) : <span className="text-xs text-slate-400">No explicit primitive list in this run.</span>}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">
              Older runs only store the selected strategy and generated attack. The planner XML is available only for newer runs.
            </p>
          )}
          {attempt?.generator.plan_raw && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-blue-600 hover:text-blue-700">Show raw plan XML</summary>
              <pre className="mt-2 text-xs whitespace-pre-wrap bg-white border border-slate-200 rounded-lg p-3 max-h-64 overflow-auto">
                {attempt.generator.plan_raw}
              </pre>
            </details>
          )}
        </div>

        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">State</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <Field label="Current strategy" value={state.current_strategy} />
            <Field label="Previous strategy" value={state.previous_strategy} />
            <Field label="Previous outcome" value={state.previous_outcome} />
            <Field label="Attempts" value={`${state.attempts_so_far}`} />
            <Field label="Leaked so far" value={state.leak_seen ? 'yes' : 'no'} />
            <Field label="Verified so far" value={state.verified_seen ? 'yes' : 'no'} />
          </div>
          <div className="mt-3 text-xs text-slate-500">
            The state is derived from run history when the runtime does not emit an explicit planner state snapshot.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
          <div className="flex items-center justify-between gap-3 mb-2">
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Best plan</h4>
            <span className="text-xs text-slate-500">{run.best_attack ? 'available' : 'missing'}</span>
          </div>
          {run.best_attack ? (
            <div className="space-y-2 text-sm">
              <Field label="Best strategy" value={run.best_attack.strategy} />
              <Field label="Best score" value={formatNumber(run.best_attack.score)} />
              <div>
                <p className="text-xs text-slate-500 mb-1">Best attack prompt</p>
                <pre className="whitespace-pre-wrap text-xs bg-white border border-slate-200 rounded-lg p-3 max-h-40 overflow-auto">
                  {run.best_attack.prompt}
                </pre>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No best plan was recorded for this run.</p>
          )}
        </div>

        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">KB / RAG usage</h4>
          <div className="space-y-2 text-sm">
            {kbRag.map((item) => (
              <div key={item.name} className="bg-white border border-slate-200 rounded-lg p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{item.name}</span>
                  <span className="text-xs text-slate-400">{item.summary}</span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{item.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
        <div className="flex items-center justify-between gap-3 mb-2">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Strategy performance</h4>
          <span className="text-xs text-slate-500">{rows.length} strategies</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs text-slate-500">
                <th className="text-left py-2 pr-3">Strategy</th>
                <th className="text-right py-2 pr-3">Successes</th>
                <th className="text-right py-2 pr-3">Partial</th>
                <th className="text-right py-2 pr-3">Failures</th>
                <th className="text-right py-2 pr-3">Rate</th>
                <th className="text-right py-2 pr-3">Score</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.strategy} className="border-b border-slate-100">
                  <td className="py-2 pr-3 font-mono text-xs">{row.strategy}</td>
                  <td className="py-2 pr-3 text-right">{row.successes}</td>
                  <td className="py-2 pr-3 text-right">{row.partial_leaks}</td>
                  <td className="py-2 pr-3 text-right">{row.failures}</td>
                  <td className="py-2 pr-3 text-right">{formatPct(row.successRate)}</td>
                  <td className="py-2 pr-3 text-right">{formatNumber(row.total_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">Timeline</h4>
        <div className="space-y-2 max-h-72 overflow-auto pr-1">
          {timeline.map((item) => (
            <div key={item.attempt_number} className={`rounded-lg border p-3 bg-white ${item.attempt_number === state.attempt_number ? 'border-blue-300 ring-1 ring-blue-200' : 'border-slate-200'}`}>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Attempt {item.attempt_number}</p>
                  <p className="text-xs text-slate-500">{item.strategy} · {item.time_ms} ms · {item.timestamp || 'n/a'}</p>
                </div>
                <span className={`text-xs font-semibold ${item.success ? 'text-green-600' : 'text-slate-500'}`}>
                  {item.success ? 'success' : 'open'}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-600">
                State: {item.state.previous_outcome}. Planner saw {item.state.success_so_far} success signal(s) in {item.state.attempts_so_far} attempt(s).
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
    <div className="bg-white border border-slate-200 rounded-lg p-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-medium text-slate-900 break-words">{value}</p>
    </div>
  );
}
