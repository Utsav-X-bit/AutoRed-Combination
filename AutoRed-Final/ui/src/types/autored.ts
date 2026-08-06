export interface ExperimentInfo {
  run_id: string;
  benchmark_mode: boolean;
  benchmark_run_number?: number;
  benchmark_total_runs?: number;
  max_attempts: number;
  dataset_size: number;
  scenario_id: string;
  seed: number;
  timestamp: string;
  experiment_version: string;
  git_commit: string;
}

export interface ModelInfo {
  name: string;
  load_time: number;
}

export interface ModelsInfo {
  victim: ModelInfo;
  generator: ModelInfo;
  judge: ModelInfo;
  extractor: ModelInfo;
}

export interface TimingInfo {
  total_run_time: number;
  model_loading_time: number;
  average_attempt_time: number;
}

export interface ScenarioInfo {
  pre_defense: string;
  post_defense: string;
  access_code: string;
  full_prompt: string;
  access_code_type?: string;
  defense_type?: string;
}

export interface ResultInfo {
  ground_truth_success: boolean;
  generator_success: boolean;
  extractor_success: boolean;
  verified_success: boolean;
  extracted_value: string;
  success_reason: "ground_truth" | "extractor" | "verification" | null;
  total_attempts: number;
}

export interface StrategyStat {
  successes: number;
  partial_leaks: number;
  failures: number;
  total_score: number;
}

export interface BestAttack {
  prompt: string;
  score: number;
  strategy: string;
}

export interface GroundTruthInfo {
  access_code: string;
  leaked: boolean;
  leak_position: number | null;
  leak_count: number;
}

export interface RankedCandidate {
  value: string;
  score: number;
}

export interface VerificationTraceItem {
  rank: number;
  candidate: string;
  score: number;
  success: boolean;
  accepted_by_victim: boolean;
  complete_match: boolean;
  victim_response: string;
}

export interface ExtractorTrace {
  regex_candidates: string[];
  quoted_candidates: string[];
  capitalized_candidates: string[];
  llm_candidates: string[];
  llm_ranked_candidates: RankedCandidate[];
  ranked_candidates: RankedCandidate[];
  top_k_candidates: RankedCandidate[];
  best_candidate: string;
  verified_candidate: string | null;
  verified_rank: number;
  verified_score: number;
  verification_response: string;
  verification_traces: VerificationTraceItem[];
}

export interface VerificationTrace {
  candidate_sent: string;
  victim_response: string;
  success: boolean;
  traces: VerificationTraceItem[];
}

export interface GeneratorInfo {
  strategy: string;
  primitives?: string[];
  style?: string;
  retry_policy?: string;
  expected_access_type?: string;
  plan_raw?: string;
  internal_prompt: string;
  generated_attack: string;
  attack_length: number;
  attack_hash: string;
  duplicate_attack: boolean;
  input_tokens: number;
  output_tokens: number;
}

export interface JudgeInfo {
  input: string;
  decision: string;
  confidence: number;
  probabilities: { ATTACK: number; ATTEMPT: number };
}

export interface VictimInfo {
  raw_output: string;
  clean_output: string;
  output_length: number;
}

export interface Attempt {
  attempt_number: number;
  timestamp: string;
  attempt_time_ms: number;
  generator: GeneratorInfo;
  judge: JudgeInfo;
  victim: VictimInfo;
  extractor: ExtractorTrace;
  verification: VerificationTrace;
  ground_truth_found: boolean;
  extractor_match: boolean;
  generator_success: boolean;
}

export interface Event {
  timestamp: string;
  type: string;
  message: string;
}

export interface SummaryStats {
  attack_length_min: number;
  attack_length_max: number;
  attack_length_avg: number;
  unique_attacks: number;
  repetition_rate: number;
  judge_distribution: { ATTACK: number; ATTEMPT: number };
}

export interface AutoRedRun {
  experiment: ExperimentInfo;
  raw_dataset_entry: Record<string, any>;
  models: ModelsInfo;
  timing: TimingInfo;
  scenario: ScenarioInfo;
  result: ResultInfo;
  strategy_stats: Record<string, StrategyStat>;
  best_attack: BestAttack | null;
  ground_truth: GroundTruthInfo;
  attempts: Attempt[];
  events: Event[];
  summary: SummaryStats;
}

export interface PlannerStateSnapshot {
  attempt_number: number;
  current_strategy: string;
  previous_strategy: string;
  previous_outcome: string;
  success_so_far: number;
  attempts_so_far: number;
  leak_seen: boolean;
  verified_seen: boolean;
}

export interface RunListItem {
  run_id: string;
  file_path: string;
  timestamp: string;
  scenario_id: string;
  success: boolean;
  total_attempts: number;
  access_code: string;
  generator: string;
  victim: string;
  benchmark_mode: boolean;
  error?: string;
}

export interface BenchmarkWorkerSummary {
  worker_id: number;
  rounds: number;
  successes: number;
  success_rate: number;
}

export interface BenchmarkMetrics {
  true_positive: number;
  false_positive: number;
  false_negative: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface BenchmarkListItem {
  benchmark_id: string;
  /** Model group for nested-layout ids ("<model>/<chars>"); null for legacy flat ids. */
  benchmark_group?: string | null;
  /** On-disk layout: "nested" (results/benchmark/<model>/<chars>) or "legacy" (results/benchmarks/<name>). */
  layout?: "nested" | "legacy";
  file_path: string;
  timestamp: string;
  total_rounds: number;
  total_successes: number;
  verified_success: number;
  success_rate: number;
  defense_rate?: number;
  total_success_exact?: number;
  total_success_extractor?: number;
  avg_attempts_on_success: number;
  top1_success: number;
  top3_success: number;
  top5_success: number;
  strategy_stats?: Record<string, any>;
  results?: Array<Record<string, any>>;
  extractor_metrics: BenchmarkMetrics;
  worker_summaries: BenchmarkWorkerSummary[];
  metadata: Record<string, any>;
  trace_archive_count: number;
  trace_archives: string[];
}

export interface TraceRunListItem {
  run_id: string;
  file_path: string;
  timestamp: string;
  scenario_id: string;
  success: boolean;
  verified_success: boolean;
  total_attempts: number;
  access_code: string;
  generator: string;
  victim: string;
  benchmark_mode: boolean;
  worker_id?: number | null;
  attempt_count: number;
}

export interface TraceArchiveDetail {
  archive_id: string;
  date: string;
  path: string;
  timestamp: string;
  run_count: number;
  success_rate: number;
  verified_rate: number;
  avg_attempts_on_success: number;
  runs: TraceRunListItem[];
}

export interface BenchmarkDetail {
  benchmark_id: string;
  benchmark_group?: string | null;
  layout?: "nested" | "legacy";
  summary: Record<string, any>;
  metadata: Record<string, any>;
  worker_summaries: BenchmarkWorkerSummary[];
  trace_archives: TraceArchiveDetail[];
  trace_runs: TraceRunListItem[];
}

export interface AttemptUpdate {
  type: "attempt_update";
  run_id: string;
  attempt: Attempt;
}

export interface RunComplete {
  type: "run_complete";
  run_id: string;
  run: AutoRedRun;
}
