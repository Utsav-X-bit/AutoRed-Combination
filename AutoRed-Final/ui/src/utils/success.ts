import type { AutoRedRun } from '../types/autored';

export function isRunSuccessful(run: Pick<AutoRedRun, 'result'>): boolean {
  const { result } = run;
  return Boolean(
    result.ground_truth_success ||
    result.extractor_success ||
    result.verified_success
  );
}
