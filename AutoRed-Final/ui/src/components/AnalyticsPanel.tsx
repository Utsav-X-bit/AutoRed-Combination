import { useRunStore } from '../store/runStore';

export default function AnalyticsPanel() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const { models, result, timing } = selectedRun;

  const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="border-b border-stone-200 p-4 last:border-b-0 dark:border-stone-800">
      <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-stone-500 dark:text-stone-400">{title}</h3>
      {children}
    </div>
  );

  const Row = ({ label, children, valueClass }: { label: string; children: React.ReactNode; valueClass?: string }) => (
    <div className="flex items-center justify-between text-sm">
      <span className="text-stone-600 dark:text-stone-400">{label}</span>
      <span className={`font-medium ${valueClass ?? 'text-stone-900 dark:text-stone-100'}`}>{children}</span>
    </div>
  );

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto border-l border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
      <Section title="Models">
        <div className="space-y-2">
          <Row label="Generator">{models.generator.name.split('/').pop()}</Row>
          <Row label="Victim">{models.victim.name.split('/').pop()}</Row>
          <Row label="Judge">DistilBERT</Row>
          <Row label="Extractor">{models.extractor.name.split('/').pop()}</Row>
        </div>
      </Section>

      <Section title="Success">
        <div className="space-y-2">
          <Row label="Ground Truth" valueClass={result.ground_truth_success ? 'font-bold text-emerald-600 dark:text-emerald-400' : 'font-bold text-rose-600 dark:text-rose-400'}>
            {result.ground_truth_success ? '✓' : '✗'}
          </Row>
          <Row label="Extractor" valueClass={result.extractor_success ? 'font-bold text-emerald-600 dark:text-emerald-400' : 'font-bold text-rose-600 dark:text-rose-400'}>
            {result.extractor_success ? '✓' : '✗'}
          </Row>
          <Row label="Verifier" valueClass={result.verified_success ? 'font-bold text-emerald-600 dark:text-emerald-400' : 'font-bold text-rose-600 dark:text-rose-400'}>
            {result.verified_success ? '✓' : '✗'}
          </Row>
        </div>
      </Section>

      <Section title="Attempts">
        <div className="space-y-2">
          <Row label="Total">{result.total_attempts}</Row>
          <Row label="Success" valueClass={result.ground_truth_success ? 'font-bold text-emerald-600 dark:text-emerald-400' : 'font-bold text-rose-600 dark:text-rose-400'}>
            {result.ground_truth_success ? 'YES' : 'NO'}
          </Row>
        </div>
      </Section>

      <Section title="Timing">
        <div className="space-y-2">
          <Row label="Total Run">{timing.total_run_time.toFixed(1)}s</Row>
          <Row label="Avg Attempt">{timing.average_attempt_time.toFixed(1)}s</Row>
        </div>
      </Section>

      <Section title="Summary">
        <div className="space-y-2">
          <Row label="Unique Attacks">{selectedRun.summary.unique_attacks}</Row>
          <Row label="Repetition">{(selectedRun.summary.repetition_rate * 100).toFixed(1)}%</Row>
          <Row label="Attack Len">{selectedRun.summary.attack_length_avg.toFixed(0)} chars</Row>
        </div>
      </Section>
    </div>
  );
}
