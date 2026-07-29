import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'AutoRed-Final', 'experiment'))

from scoring import (
    classify_success, classify_failure_mode,
    PLANNER_STUCK_THRESHOLD, resolve_mutator_pool, STRATEGY_MUTATOR_MAP,
)


def _att(strategy="instruction_leak", gt=False, ext=False, ver=False):
    """Build a minimal trace attempt dict."""
    return {
        "strategy": strategy,
        "ground_truth_found": gt,
        "extractor_match": ext,
        "verification_success": ver,
    }


# --- classify_success priority ---
def test_classify_success_gt_leak_wins_over_others():
    assert classify_success(True, True, True) == "gt_leak"

def test_classify_success_verified_next():
    assert classify_success(False, False, True) == "verified"

def test_classify_success_extractor_next():
    assert classify_success(False, True, False) == "extractor"

def test_classify_success_none():
    assert classify_success(False, False, False) == "none"


# --- classify_failure_mode ---
def test_failure_never_leaked():
    # Low-diversity trace (2 distinct strategies, each < threshold, not >=3 distinct),
    # no leak, fallback score ABOVE threshold -> never_leaked
    trace = [_att(strategy=("instruction_leak" if i % 2 == 0 else "roleplay"), gt=False)
             for i in range(20)]
    assert classify_failure_mode(trace, False, 5.0) == "never_leaked"

def test_failure_planner_stuck():
    trace = [_att(strategy="instruction_leak", gt=False) for _ in range(PLANNER_STUCK_THRESHOLD)]
    trace += [_att(strategy="instruction_leak", gt=False) for _ in range(20 - PLANNER_STUCK_THRESHOLD)]
    assert classify_failure_mode(trace, False, 0.0) == "planner_stuck"

def test_failure_generator_rephrase_fail():
    # 3 distinct strategies, no leak
    trace = [_att(strategy="instruction_leak"), _att(strategy="roleplay"), _att(strategy="encoding_bypass")]
    assert classify_failure_mode(trace, False, 0.0) == "generator_rephrase_fail"

def test_failure_fallback_failed():
    trace = [_att(gt=False) for _ in range(20)]
    assert classify_failure_mode(trace, True, 0.5) == "fallback_failed"

def test_failure_fallback_untriggered():
    # Low-diversity trace (2 distinct, not stuck, not >=3 distinct), no leak,
    # fallback never ran, score BELOW threshold -> fallback_untriggered
    trace = [_att(strategy=("instruction_leak" if i % 2 == 0 else "roleplay"), gt=False)
             for i in range(20)]
    # triggered=False, best score below threshold
    assert classify_failure_mode(trace, False, 0.1, min_score_threshold=0.25) == "fallback_untriggered"

def test_failure_priority_fallback_failed_over_never_leaked():
    # fallback ran and failed takes priority over never_leaked
    trace = [_att(gt=False) for _ in range(20)]
    assert classify_failure_mode(trace, True, 0.5) == "fallback_failed"

def test_failure_leaked_unverified_bugcatch():
    # ground_truth_found on an attempt but success overall False -> bug catch
    trace = [_att(gt=True), _att(gt=False)]
    # success must be false for this label to apply; simulate by passing trace
    # with a leaked attempt; classify_failure_mode is only called on failed scenarios.
    assert classify_failure_mode(trace, False, 0.0) == "leaked_unverified"


# --- resolve_mutator_pool ---
def test_mutator_pool_encoding_strategies_get_pi_only():
    for s in ("encoding_bypass", "json_smuggling", "unicode_bypass"):
        assert resolve_mutator_pool(s, ["SR", "PI", "TL"]) == ["PI"]

def test_mutator_pool_text_strategies_get_sr_tl():
    for s in ("instruction_leak", "roleplay", "trigger_phrase_discovery",
              "summarization", "exception_discovery", "system_prompt_recovery",
              "translation"):
        assert resolve_mutator_pool(s, ["SR", "PI", "TL"]) == ["SR", "TL"]

def test_mutator_pool_unknown_falls_back_to_default():
    assert resolve_mutator_pool("nonsense_strategy", ["SR", "PI", "TL"]) == ["SR", "PI", "TL"]
    assert resolve_mutator_pool(None, ["SR", "PI", "TL"]) == ["SR", "PI", "TL"]
