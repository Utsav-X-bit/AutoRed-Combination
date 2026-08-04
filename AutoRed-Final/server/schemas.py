from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class SuccessReason(str, Enum):
    ground_truth = "ground_truth"
    extractor = "extractor"
    verification = "verification"


class ExperimentInfo(BaseModel):
    run_id: str
    benchmark_mode: bool
    benchmark_run_number: Optional[int] = None
    benchmark_total_runs: Optional[int] = None
    max_attempts: int
    dataset_size: int
    scenario_id: str
    seed: int
    timestamp: str
    experiment_version: str
    git_commit: str


class ModelInfo(BaseModel):
    name: str
    load_time: float


class ModelsInfo(BaseModel):
    victim: ModelInfo
    generator: ModelInfo
    judge: ModelInfo
    extractor: ModelInfo


class TimingInfo(BaseModel):
    total_run_time: float
    model_loading_time: float
    average_attempt_time: float


class ScenarioInfo(BaseModel):
    pre_defense: str
    post_defense: str
    access_code: str
    full_prompt: str


class ResultInfo(BaseModel):
    ground_truth_success: bool
    generator_success: bool
    extractor_success: bool
    verified_success: bool
    extracted_value: str
    success_reason: Optional[SuccessReason] = None
    total_attempts: int


class StrategyStat(BaseModel):
    successes: int = 0
    partial_leaks: int = 0
    failures: int = 0
    total_score: float = 0.0


class BestAttack(BaseModel):
    prompt: str
    score: float
    strategy: str


class GroundTruthInfo(BaseModel):
    access_code: str
    leaked: bool
    leak_position: Optional[int] = None
    leak_count: int


class RankedCandidate(BaseModel):
    value: str
    score: float


class VerificationTraceItem(BaseModel):
    rank: int
    candidate: str
    score: float
    success: bool
    accepted_by_victim: bool = False
    complete_match: bool = False
    victim_response: str = ""


class ExtractorTrace(BaseModel):
    regex_candidates: List[str] = Field(default_factory=list)
    quoted_candidates: List[str] = Field(default_factory=list)
    capitalized_candidates: List[str] = Field(default_factory=list)
    llm_candidates: List[str] = Field(default_factory=list)
    llm_ranked_candidates: List[RankedCandidate] = Field(default_factory=list)
    ranked_candidates: List[RankedCandidate] = Field(default_factory=list)
    top_k_candidates: List[RankedCandidate] = Field(default_factory=list)
    best_candidate: str = ""
    verified_candidate: str = ""
    verified_rank: int = 0
    verified_score: float = 0.0
    verification_response: str = ""
    verification_traces: List[VerificationTraceItem] = Field(default_factory=list)


class VerificationTrace(BaseModel):
    candidate_sent: str = ""
    victim_response: str = ""
    success: bool = False
    traces: List[VerificationTraceItem] = Field(default_factory=list)


class GeneratorInfo(BaseModel):
    strategy: str
    internal_prompt: str
    generated_attack: str
    attack_length: int
    attack_hash: str
    duplicate_attack: bool
    input_tokens: int
    output_tokens: int


class JudgeInfo(BaseModel):
    input: str
    decision: str
    confidence: float
    probabilities: Dict[str, float]


class VictimInfo(BaseModel):
    raw_output: str
    clean_output: str
    output_length: int


class Attempt(BaseModel):
    attempt_number: int
    timestamp: str
    attempt_time_ms: int
    generator: GeneratorInfo
    judge: JudgeInfo
    victim: VictimInfo
    extractor: ExtractorTrace
    verification: VerificationTrace
    ground_truth_found: bool
    extractor_match: bool
    generator_success: bool


class Event(BaseModel):
    timestamp: str
    type: str
    message: str


class SummaryStats(BaseModel):
    attack_length_min: int
    attack_length_max: int
    attack_length_avg: float
    unique_attacks: int
    repetition_rate: float
    judge_distribution: Dict[str, int]


class AutoRedRun(BaseModel):
    experiment: ExperimentInfo
    raw_dataset_entry: Dict[str, Any]
    models: ModelsInfo
    timing: TimingInfo
    scenario: ScenarioInfo
    result: ResultInfo
    strategy_stats: Dict[str, StrategyStat]
    best_attack: Optional[BestAttack] = None
    ground_truth: GroundTruthInfo
    attempts: List[Attempt]
    events: List[Event]
    summary: SummaryStats


# WebSocket messages
class AttemptUpdate(BaseModel):
    type: str = "attempt_update"
    run_id: str
    attempt: Attempt


class RunComplete(BaseModel):
    type: str = "run_complete"
    run_id: str
    run: AutoRedRun
