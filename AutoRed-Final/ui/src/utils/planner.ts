import type { AutoRedRun, Attempt, PlannerStateSnapshot, StrategyStat } from '../types/autored';

export interface ParsedPlan {
  strategy: string;
  primitives: string[];
  style: string;
  expected_access_type: string;
  retry_policy: string;
  confidence: number;
  failure_reason: string;
}

const tag = (text: string, name: string) => {
  const match = text.match(new RegExp(`<${name}>([\\s\\S]*?)</${name}>`, 'i'));
  return match ? match[1].trim() : '';
};

export function parsePlanText(planRaw: string | undefined | null): ParsedPlan | null {
  if (!planRaw) return null;
  const primitiveBlock = tag(planRaw, 'primitive_sequence');
  const primitives = Array.from(primitiveBlock.matchAll(/<step>(.*?)<\/step>/gi))
    .map((match) => match[1].trim())
    .filter(Boolean);

  const confidenceRaw = tag(planRaw, 'confidence');
  const confidence = Number(confidenceRaw);

  return {
    strategy: tag(planRaw, 'strategy') || 'unknown',
    primitives: primitives.length ? primitives : [],
    style: tag(planRaw, 'style') || 'unknown',
    expected_access_type: tag(planRaw, 'expected_access_type')
      || tag(planRaw, 'expected_access_code_type')
      || 'UNKNOWN',
    retry_policy: tag(planRaw, 'retry_policy') || 'explore',
    confidence: Number.isFinite(confidence) ? confidence : -1,
    failure_reason: tag(planRaw, 'failure_reason') || 'none',
  };
}

export function isNewPlannerRun(run: AutoRedRun): boolean {
  return run.attempts.some((attempt) => Boolean(attempt.generator.plan_raw));
}

export function buildPlannerState(run: AutoRedRun, attemptIndex: number): PlannerStateSnapshot {
  const attempt = run.attempts[attemptIndex];
  const previous = run.attempts[attemptIndex - 1];
  const priorAttempts = run.attempts.slice(0, attemptIndex + 1);
  return {
    attempt_number: attempt?.attempt_number ?? attemptIndex + 1,
    current_strategy: attempt?.generator.strategy ?? 'unknown',
    previous_strategy: previous?.generator.strategy ?? 'n/a',
    previous_outcome: previous
      ? previous.extractor_match
        ? 'extractor success'
        : previous.ground_truth_found
          ? 'ground truth leaked'
          : previous.verification.success
            ? 'verification success'
            : 'no leak'
      : 'start of run',
    success_so_far: priorAttempts.filter((item) => item.ground_truth_found || item.extractor_match || item.verification.success).length,
    attempts_so_far: priorAttempts.length,
    leak_seen: priorAttempts.some((item) => item.ground_truth_found),
    verified_seen: priorAttempts.some((item) => item.verification.success),
  };
}

export function strategyRows(strategyStats: Record<string, StrategyStat>) {
  return Object.entries(strategyStats)
    .map(([strategy, stat]) => {
      const attempts = stat.successes + stat.partial_leaks + stat.failures;
      const successRate = attempts ? stat.successes / attempts : 0;
      return { strategy, ...stat, attempts, successRate };
    })
    .sort((a, b) => b.successRate - a.successRate || b.total_score - a.total_score);
}

export function attemptTimeline(run: AutoRedRun) {
  return run.attempts.map((attempt, index) => ({
    attempt_number: attempt.attempt_number,
    strategy: attempt.generator.strategy,
    plan_raw: attempt.generator.plan_raw ?? '',
    parsed_plan: parsePlanText(attempt.generator.plan_raw),
    time_ms: attempt.attempt_time_ms,
    timestamp: attempt.timestamp,
    success: attempt.ground_truth_found || attempt.extractor_match || attempt.verification.success,
    state: buildPlannerState(run, index),
  }));
}

export function kbRagSummary(run: AutoRedRun) {
  const raw = run.raw_dataset_entry ?? {};
  return [
    {
      name: 'KB',
      summary: 'Historical strategy statistics and best attack memory',
      value: `${Object.keys(run.strategy_stats).length} strategy buckets, best attack: ${run.best_attack?.strategy ?? 'n/a'}`,
    },
    {
      name: 'RAG',
      summary: 'Retrieved scenario and history context for the planner',
      value: `scenario ${run.experiment.scenario_id}, ${run.attempts.length} prior attempts, ${raw.access_code_type ?? 'unknown'} access code type`,
    },
    {
      name: 'Oracle',
      summary: 'Hindsight target used by the dataset tooling',
      value: `defense_id ${raw.defense_id ?? 'n/a'}, defense_type ${raw.defense_type ?? 'n/a'}`,
    },
  ];
}
