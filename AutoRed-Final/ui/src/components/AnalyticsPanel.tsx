import { useRunStore } from '../store/runStore';

export default function AnalyticsPanel() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const { models, result, timing } = selectedRun;

  return (
    <div className="w-full bg-white flex flex-col h-full overflow-y-auto">
      {/* Models */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Models</h3>
        <div className="space-y-2">
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Generator</p>
            <p className="font-medium text-slate-900 truncate" title={models.generator.name}>{models.generator.name.split('/').pop()}</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Victim</p>
            <p className="font-medium text-slate-900 truncate" title={models.victim.name}>{models.victim.name.split('/').pop()}</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Judge</p>
            <p className="font-medium text-slate-900 truncate" title={models.judge.name}>DistilBERT</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Extractor</p>
            <p className="font-medium text-slate-900 truncate" title={models.extractor.name}>{models.extractor.name.split('/').pop()}</p>
          </div>
        </div>
      </div>

      {/* Success Metrics */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Success</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Ground Truth</span>
            <span className={`font-bold ${result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.ground_truth_success ? '✓' : '✗'}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Extractor</span>
            <span className={`font-bold ${result.extractor_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.extractor_success ? '✓' : '✗'}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Verifier</span>
            <span className={`font-bold ${result.verified_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.verified_success ? '✓' : '✗'}
            </span>
          </div>
        </div>
      </div>

      {/* Attempts */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Attempts</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Total</span>
            <span className="font-bold text-slate-900">{result.total_attempts}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Success</span>
            <span className={`font-bold ${result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.ground_truth_success ? 'YES' : 'NO'}
            </span>
          </div>
        </div>
      </div>

      {/* Timing */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Timing</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Total Run</span>
            <span className="font-medium text-slate-900">{timing.total_run_time.toFixed(1)}s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Avg Attempt</span>
            <span className="font-medium text-slate-900">{timing.average_attempt_time.toFixed(1)}s</span>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="p-4">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Summary</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Unique Attacks</span>
            <span className="font-medium text-slate-900">{selectedRun.summary.unique_attacks}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Repetition</span>
            <span className="font-medium text-slate-900">{(selectedRun.summary.repetition_rate * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Attack Len</span>
            <span className="font-medium text-slate-900">{selectedRun.summary.attack_length_avg.toFixed(0)} chars</span>
          </div>
        </div>
      </div>
    </div>
  );
}
