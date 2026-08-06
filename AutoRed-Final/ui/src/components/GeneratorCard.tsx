import { Attempt } from '../types/autored';
import { parsePlanText } from '../utils/planner';

export default function GeneratorCard({ attempt }: { attempt: Attempt }) {
  const parsedPlan = parsePlanText(attempt.generator.plan_raw);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🧠</span> Generator
        </h3>
        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
          {attempt.generator.strategy}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Input Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.input_tokens}</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Output Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.output_tokens}</p>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 mb-1.5">Generated Attack</p>
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-900 font-mono leading-relaxed">
            "{attempt.generator.generated_attack}"
          </p>
        </div>
      </div>

      <div className="mt-4 border-t border-slate-200 pt-4">
        <div className="flex items-center justify-between gap-3 mb-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Planner contract</p>
          <span className="text-xs text-slate-400">
            {attempt.generator.plan_raw ? 'new pipeline' : 'legacy run'}
          </span>
        </div>

        {parsedPlan ? (
          <div className="grid grid-cols-2 gap-2 text-sm">
            <Meta label="Strategy" value={parsedPlan.strategy} />
            <Meta label="Style" value={parsedPlan.style} />
            <Meta label="Access type" value={parsedPlan.expected_access_type} />
            <Meta label="Retry" value={parsedPlan.retry_policy} />
            <Meta label="Confidence" value={parsedPlan.confidence >= 0 ? parsedPlan.confidence.toFixed(2) : 'n/a'} />
            <Meta label="Failure reason" value={parsedPlan.failure_reason} />
            <div className="col-span-2">
              <p className="text-xs text-slate-500 mb-1">Primitives</p>
              <div className="flex flex-wrap gap-1.5">
                {parsedPlan.primitives.length ? parsedPlan.primitives.map((primitive) => (
                  <span key={primitive} className="px-2 py-1 bg-slate-50 border border-slate-200 rounded-md text-xs text-slate-700">
                    {primitive}
                  </span>
                )) : <span className="text-xs text-slate-400">No explicit primitives in this run.</span>}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            This run does not store the planner XML. The attack was generated from the legacy prompt path.
          </p>
        )}

        {attempt.generator.plan_raw && (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs text-blue-600 hover:text-blue-700">Show raw planner XML</summary>
            <pre className="mt-2 text-xs whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200 max-h-64 overflow-y-auto">
              {attempt.generator.plan_raw}
            </pre>
          </details>
        )}
      </div>

      {attempt.generator.duplicate_attack && (
        <div className="mt-2 px-2 py-1 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700 font-medium">
          ⚠️ Duplicate attack detected
        </div>
      )}
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-medium text-slate-900 break-words">{value}</p>
    </div>
  );
}
