import { useRunStore } from '../store/runStore';

export default function ScenarioTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const { scenario, raw_dataset_entry } = selectedRun;

  return (
    <div className="space-y-4 pb-8">
      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Scenario ID</h3>
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-lg border border-teal-200 bg-teal-50 px-4 py-2 font-mono text-lg font-bold text-teal-700 dark:border-teal-900 dark:bg-teal-950/30 dark:text-teal-400">
            {selectedRun.experiment.scenario_id}
          </span>
          <span className="text-xs text-stone-500 dark:text-stone-400">dataset defense id</span>
        </div>
      </section>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Defense Scenario</h3>
        <div className="space-y-3">
          <div>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Pre-Defense (Opening)</p>
            <details className="group">
              <summary className="cursor-pointer text-sm text-stone-700 hover:text-stone-900 dark:text-stone-300 dark:hover:text-stone-100">
                Click to expand ({scenario.pre_defense.length} chars)
              </summary>
              <pre className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-stone-950 p-3 text-sm text-stone-100 dark:border-stone-800">
                {scenario.pre_defense}
              </pre>
            </details>
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Post-Defense (Closing)</p>
            <details className="group">
              <summary className="cursor-pointer text-sm text-stone-700 hover:text-stone-900 dark:text-stone-300 dark:hover:text-stone-100">
                Click to expand ({scenario.post_defense.length} chars)
              </summary>
              <pre className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-stone-950 p-3 text-sm text-stone-100 dark:border-stone-800">
                {scenario.post_defense}
              </pre>
            </details>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Access Code</h3>
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 font-mono text-lg font-bold text-amber-700 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-400">
            {scenario.access_code}
          </span>
          <span className="text-xs text-stone-500 dark:text-stone-400">({scenario.access_code.length} chars)</span>
        </div>
      </section>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Raw Dataset Entry</h3>
        <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap rounded-lg border border-stone-200 bg-stone-950 p-3 text-xs text-stone-100 dark:border-stone-800">
          {JSON.stringify(raw_dataset_entry, null, 2)}
        </pre>
        <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Defense Type</p>
            <p className="font-medium text-stone-900 dark:text-stone-100">{raw_dataset_entry.defense_type ?? 'n/a'}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Access Code Type</p>
            <p className="font-medium text-stone-900 dark:text-stone-100">{raw_dataset_entry.access_code_type ?? 'n/a'}</p>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm dark:border-stone-800 dark:bg-stone-900">
        <h3 className="mb-3 font-display text-sm font-semibold text-stone-900 dark:text-stone-100">Experiment Metadata</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Version</p>
            <p className="font-mono font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.experiment_version}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Git Commit</p>
            <p className="break-all font-mono text-xs font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.git_commit}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Scenario ID</p>
            <p className="font-mono font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.scenario_id}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Seed</p>
            <p className="font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.seed}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Benchmark Mode</p>
            <p className="font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.benchmark_mode ? 'YES' : 'NO'}</p>
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">Max Attempts</p>
            <p className="font-medium text-stone-900 dark:text-stone-100">{selectedRun.experiment.max_attempts}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
