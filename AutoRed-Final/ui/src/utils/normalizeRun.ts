import type {
  Attempt,
  AutoRedRun,
  RankedCandidate,
  RunListItem,
} from '../types/autored';

const asObject = (value: unknown): Record<string, any> =>
  value && typeof value === 'object' ? value as Record<string, any> : {};
const asArray = (value: unknown): any[] => Array.isArray(value) ? value : [];
const asString = (value: unknown, fallback = ''): string =>
  value == null ? fallback : String(value);
const asNumber = (value: unknown, fallback = 0): number => {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const normalizeRankedCandidates = (value: unknown): RankedCandidate[] =>
  asArray(value).flatMap((candidate) => {
    if (Array.isArray(candidate)) {
      const item = { value: asString(candidate[0]), score: asNumber(candidate[1]) };
      return item.value ? [item] : [];
    }
    const raw = asObject(candidate);
    const item = { value: asString(raw.value), score: asNumber(raw.score) };
    return item.value ? [item] : [];
  });

export function normalizeAttempt(value: unknown, index = 0): Attempt {
  const raw = asObject(value);
  const generator = asObject(raw.generator);
  const judge = asObject(raw.judge);
  const victim = asObject(raw.victim);
  const extractor = asObject(raw.extractor);
  const verification = asObject(raw.verification);
  const probabilities = asObject(judge.probabilities);
  const attack = asString(generator.generated_attack);
  const response = asString(victim.raw_output);
  const extractorVerificationTraces = (asArray(extractor.verification_traces) || []).map((t: any) => ({
    rank: Number(t.rank) || 0,
    candidate: String(t.candidate ?? ''),
    score: Number(t.score) || 0,
    success: Boolean(t.success),
    accepted_by_victim: Boolean(t.accepted_by_victim),
    complete_match: Boolean(t.complete_match),
    victim_response: asString(t.victim_response),
  }));
  const verificationTraces = (asArray(verification.traces) || []).map((t: any) => ({
    rank: Number(t.rank) || 0,
    candidate: String(t.candidate ?? ''),
    score: Number(t.score) || 0,
    success: Boolean(t.success),
    accepted_by_victim: Boolean(t.accepted_by_victim),
    complete_match: Boolean(t.complete_match),
    victim_response: asString(t.victim_response),
  }));
  const traceResponse = (
    verificationTraces.find(t => t.success)?.victim_response
    || extractorVerificationTraces.find(t => t.success)?.victim_response
    || verificationTraces[verificationTraces.length - 1]?.victim_response
    || extractorVerificationTraces[extractorVerificationTraces.length - 1]?.victim_response
    || ''
  );

  return {
    attempt_number: asNumber(raw.attempt_number, index + 1),
    timestamp: asString(raw.timestamp),
    attempt_time_ms: asNumber(raw.attempt_time_ms),
    generator: {
      strategy: asString(generator.strategy, 'unknown'),
      primitives: asArray(generator.primitives).map((value) => asString(value)).filter(Boolean),
      style: asString(generator.style, 'unknown'),
      retry_policy: asString(generator.retry_policy, 'explore'),
      expected_access_type: asString(
        generator.expected_access_type,
        asString(generator.expected_access_code_type, 'UNKNOWN'),
      ),
      plan_raw: asString(generator.plan_raw),
      internal_prompt: asString(generator.internal_prompt),
      generated_attack: attack,
      attack_length: asNumber(generator.attack_length, attack.length),
      attack_hash: asString(generator.attack_hash),
      duplicate_attack: Boolean(generator.duplicate_attack),
      input_tokens: asNumber(generator.input_tokens),
      output_tokens: asNumber(generator.output_tokens),
    },
    judge: {
      input: asString(judge.input),
      decision: asString(judge.decision),
      confidence: asNumber(judge.confidence),
      probabilities: {
        ATTACK: asNumber(probabilities.ATTACK ?? probabilities['ATTACK (0)']),
        ATTEMPT: asNumber(probabilities.ATTEMPT ?? probabilities['ATTEMPT (1)']),
      },
    },
    victim: {
      raw_output: response,
      clean_output: asString(victim.clean_output, response),
      output_length: asNumber(victim.output_length, response.length),
    },
    extractor: {
      regex_candidates: asArray(extractor.regex_candidates).map(String),
      quoted_candidates: asArray(extractor.quoted_candidates).map(String),
      capitalized_candidates: asArray(extractor.capitalized_candidates).map(String),
      llm_candidates: asArray(extractor.llm_candidates).map(String),
      llm_ranked_candidates: normalizeRankedCandidates(extractor.llm_ranked_candidates),
      ranked_candidates: normalizeRankedCandidates(extractor.ranked_candidates),
      top_k_candidates: normalizeRankedCandidates(extractor.top_k_candidates),
      best_candidate: asString(extractor.best_candidate),
      verified_candidate: extractor.verified_candidate ?? null,
      verified_rank: Number(extractor.verified_rank) || 0,
      verified_score: Number(extractor.verified_score) || 0,
      verification_response: asString(extractor.verification_response),
      verification_traces: extractorVerificationTraces,
    },
    verification: {
      candidate_sent: asString(verification.candidate_sent),
      victim_response: asString(verification.victim_response) || asString(extractor.verification_response) || traceResponse,
      success: Boolean(verification.success),
      traces: verificationTraces,
    },
    ground_truth_found: Boolean(raw.ground_truth_found),
    extractor_match: Boolean(raw.extractor_match),
    generator_success: Boolean(raw.generator_success),
  };
}

export function normalizeRun(value: unknown): AutoRedRun {
  const raw = asObject(value);
  const experiment = asObject(raw.experiment);
  const timing = asObject(raw.timing);
  const scenario = asObject(raw.scenario);
  const result = asObject(raw.result);
  const groundTruth = asObject(raw.ground_truth);
  const rawDatasetEntry = asObject(raw.raw_dataset_entry);
  const attempts = asArray(raw.attempts).map(normalizeAttempt);
  const lengths = attempts.map((attempt) => attempt.generator.attack_length);
  const uniqueAttacks = new Set(
    attempts.map((attempt) => attempt.generator.generated_attack),
  ).size;
  const summary = asObject(raw.summary);
  const judgeDistribution = attempts.reduce(
    (counts, attempt) => {
      if (attempt.judge.decision === 'ATTACK') counts.ATTACK += 1;
      if (attempt.judge.decision === 'ATTEMPT') counts.ATTEMPT += 1;
      return counts;
    },
    { ATTACK: 0, ATTEMPT: 0 },
  );
  const models = asObject(raw.models);
  const model = (name: string) => {
    const item = asObject(models[name]);
    return { name: asString(item.name, 'unknown'), load_time: asNumber(item.load_time) };
  };
  const strategyStats = Object.fromEntries(
    Object.entries(asObject(raw.strategy_stats)).map(([name, value]) => {
      const stat = asObject(value);
      return [name, {
        successes: asNumber(stat.successes),
        partial_leaks: asNumber(stat.partial_leaks),
        failures: asNumber(stat.failures),
        total_score: asNumber(stat.total_score),
      }];
    }),
  );

  return {
    experiment: {
      run_id: asString(experiment.run_id, 'unknown-run'),
      benchmark_mode: Boolean(experiment.benchmark_mode),
      benchmark_run_number: experiment.benchmark_run_number == null
        ? undefined : asNumber(experiment.benchmark_run_number),
      benchmark_total_runs: experiment.benchmark_total_runs == null
        ? undefined : asNumber(experiment.benchmark_total_runs),
      max_attempts: asNumber(experiment.max_attempts, attempts.length),
      dataset_size: asNumber(experiment.dataset_size),
      scenario_id: asString(experiment.scenario_id, asString(rawDatasetEntry.defense_id, 'unknown')),
      seed: asNumber(experiment.seed, 42),
      timestamp: asString(experiment.timestamp),
      experiment_version: asString(experiment.experiment_version, 'unknown'),
      git_commit: asString(experiment.git_commit, 'unknown'),
    },
    raw_dataset_entry: rawDatasetEntry,
    models: {
      victim: model('victim'),
      generator: model('generator'),
      judge: model('judge'),
      extractor: model('extractor'),
    },
    timing: {
      total_run_time: asNumber(timing.total_run_time),
      model_loading_time: asNumber(timing.model_loading_time),
      average_attempt_time: asNumber(timing.average_attempt_time),
    },
    scenario: {
      pre_defense: asString(scenario.pre_defense),
      post_defense: asString(scenario.post_defense),
      access_code: asString(scenario.access_code),
      full_prompt: asString(scenario.full_prompt),
      access_code_type: asString(
        scenario.access_code_type,
        asString(rawDatasetEntry.access_code_type, 'UNKNOWN'),
      ),
      defense_type: asString(
        scenario.defense_type,
        asString(rawDatasetEntry.defense_type, 'UNKNOWN'),
      ),
    },
    result: {
      ground_truth_success: Boolean(result.ground_truth_success),
      generator_success: Boolean(result.generator_success),
      extractor_success: Boolean(result.extractor_success),
      verified_success: Boolean(result.verified_success),
      extracted_value: asString(result.extracted_value),
      success_reason: ['ground_truth', 'extractor', 'verification'].includes(result.success_reason)
        ? result.success_reason : null,
      total_attempts: asNumber(result.total_attempts, attempts.length),
    },
    strategy_stats: strategyStats,
    best_attack: raw.best_attack ? {
      prompt: asString(asObject(raw.best_attack).prompt),
      score: asNumber(asObject(raw.best_attack).score),
      strategy: asString(asObject(raw.best_attack).strategy, 'unknown'),
    } : null,
    ground_truth: {
      access_code: asString(groundTruth.access_code, asString(scenario.access_code)),
      leaked: Boolean(groundTruth.leaked),
      leak_position: groundTruth.leak_position == null
        ? null : asNumber(groundTruth.leak_position),
      leak_count: asNumber(groundTruth.leak_count),
    },
    attempts,
    events: asArray(raw.events).map((event) => ({
      timestamp: asString(asObject(event).timestamp),
      type: asString(asObject(event).type),
      message: asString(asObject(event).message),
    })),
    summary: {
      attack_length_min: asNumber(
        summary.attack_length_min,
        lengths.length ? Math.min(...lengths) : 0,
      ),
      attack_length_max: asNumber(
        summary.attack_length_max,
        lengths.length ? Math.max(...lengths) : 0,
      ),
      attack_length_avg: asNumber(
        summary.attack_length_avg,
        lengths.length ? lengths.reduce((sum, length) => sum + length, 0) / lengths.length : 0,
      ),
      unique_attacks: asNumber(summary.unique_attacks, uniqueAttacks),
      repetition_rate: asNumber(
        summary.repetition_rate,
        attempts.length ? (attempts.length - uniqueAttacks) / attempts.length : 0,
      ),
      judge_distribution: {
        ATTACK: asNumber(asObject(summary.judge_distribution).ATTACK, judgeDistribution.ATTACK),
        ATTEMPT: asNumber(asObject(summary.judge_distribution).ATTEMPT, judgeDistribution.ATTEMPT),
      },
    },
  };
}

export function normalizeRunList(value: unknown): RunListItem[] {
  return asArray(value).map((item) => {
    const raw = asObject(item);
    return {
      run_id: asString(raw.run_id),
      file_path: asString(raw.file_path),
      timestamp: asString(raw.timestamp),
      scenario_id: asString(raw.scenario_id, 'unknown'),
      success: Boolean(raw.success),
      total_attempts: asNumber(raw.total_attempts),
      access_code: asString(raw.access_code),
      generator: asString(raw.generator),
      victim: asString(raw.victim),
      benchmark_mode: Boolean(raw.benchmark_mode),
      error: raw.error ? asString(raw.error) : undefined,
    };
  });
}
