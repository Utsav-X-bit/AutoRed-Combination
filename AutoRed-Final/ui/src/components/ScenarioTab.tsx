import { useRunStore } from '../store/runStore';

export default function ScenarioTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const { scenario, raw_dataset_entry } = selectedRun;

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Scenario ID</h3>
        <div className="flex items-center gap-3">
          <span className="font-mono text-lg font-bold text-blue-700 bg-blue-50 px-4 py-2 rounded-lg border border-blue-200">
            {selectedRun.experiment.scenario_id}
          </span>
          <span className="text-xs text-slate-500">dataset defense_id</span>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Defense Scenario</h3>
        <div className="space-y-3">
          <div>
            <p className="text-xs text-slate-500 mb-1">Pre-Defense (Opening)</p>
            <details className="group">
              <summary className="text-sm text-slate-700 cursor-pointer hover:text-slate-900">
                Click to expand ({scenario.pre_defense.length} chars)
              </summary>
              <pre className="mt-2 text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200 max-h-64 overflow-y-auto">
                {scenario.pre_defense}
              </pre>
            </details>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-1">Post-Defense (Closing)</p>
            <details className="group">
              <summary className="text-sm text-slate-700 cursor-pointer hover:text-slate-900">
                Click to expand ({scenario.post_defense.length} chars)
              </summary>
              <pre className="mt-2 text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200 max-h-64 overflow-y-auto">
                {scenario.post_defense}
              </pre>
            </details>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Access Code</h3>
        <div className="flex items-center gap-3">
          <span className="font-mono text-lg font-bold text-amber-600 bg-amber-50 px-4 py-2 rounded-lg border border-amber-200">
            {scenario.access_code}
          </span>
          <span className="text-xs text-slate-500">({scenario.access_code.length} chars)</span>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Raw Dataset Entry</h3>
        <pre className="text-xs text-slate-700 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200 max-h-48 overflow-y-auto">
          {JSON.stringify(raw_dataset_entry, null, 2)}
        </pre>
        <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-xs text-slate-500">Defense Type</p>
            <p className="font-medium">{raw_dataset_entry.defense_type ?? 'n/a'}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Access Code Type</p>
            <p className="font-medium">{raw_dataset_entry.access_code_type ?? 'n/a'}</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Experiment Metadata</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-xs text-slate-500">Version</p>
            <p className="font-mono font-medium">{selectedRun.experiment.experiment_version}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Git Commit</p>
            <p className="font-mono font-medium text-xs">{selectedRun.experiment.git_commit}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Scenario ID</p>
            <p className="font-mono font-medium">{selectedRun.experiment.scenario_id}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Seed</p>
            <p className="font-medium">{selectedRun.experiment.seed}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Benchmark Mode</p>
            <p className="font-medium">{selectedRun.experiment.benchmark_mode ? 'YES' : 'NO'}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Max Attempts</p>
            <p className="font-medium">{selectedRun.experiment.max_attempts}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
