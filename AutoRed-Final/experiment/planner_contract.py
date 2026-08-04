"""
Shared planner output contract utilities.

This module centralizes parsing, normalization, validation, and canonicalization
for the Planner's structured <plan> output so isolation tests and future
runtime integration use the same rules.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List


KNOWN_STRATEGIES = [
    "instruction_leak",
    "trigger_phrase_discovery",
    "exception_discovery",
    "roleplay",
    "summarization",
    "translation",
    "system_prompt_recovery",
    "encoding_bypass",
    "jailbreak_framing",
    "authority_override",
    "reflection_attack",
    "format_conversion",
    "base64_bypass",
    "unicode_bypass",
    "latent_injection",
    "markdown_smuggling",
    "json_smuggling",
    "yaml_smuggling",
]
KNOWN_STYLES = ["formal", "conversational", "academic", "story", "direct"]
KNOWN_POLICIES = ["explore", "retry_same_strategy", "switch_strategy"]
KNOWN_ACCESS_TYPES = ["TOKEN", "PHRASE", "SENTENCE", "MULTILINE", "UNKNOWN"]
KNOWN_FAILURE_REASONS = ["none", "JUDGE_REJECT", "EXTRACTOR_MISS", "VERIFIER_REJECT", "NEAR_MISS", "NO_RESPONSE"]


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_plan_text(output: str) -> Dict[str, Any]:
    confidence_raw = extract_tag(output, "confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else -1.0
    except (TypeError, ValueError):
        confidence = -1.0

    prim_block = extract_tag(output, "primitive_sequence") or ""
    primitives = re.findall(r"<step>(.*?)</step>", prim_block, re.DOTALL)
    if not primitives and prim_block.strip():
        try:
            parsed = json.loads(prim_block)
            if isinstance(parsed, list):
                primitives = [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    return {
        "strategy": extract_tag(output, "strategy"),
        "primitives": [p.strip() for p in primitives if p.strip()],
        "style": extract_tag(output, "style"),
        "expected_access_type": extract_tag(output, "expected_access_type"),
        "expected_access_code_type": extract_tag(output, "expected_access_code_type"),
        "expected_access_code": extract_tag(output, "expected_access_code"),
        "expected_output": extract_tag(output, "expected_output"),
        "retry_policy": extract_tag(output, "retry_policy"),
        "confidence": confidence,
        "failure_reason": extract_tag(output, "failure_reason"),
    }


def normalize_plan_dict(plan: Dict[str, Any], output: str) -> Dict[str, Any]:
    normalized = dict(plan)

    if normalized["expected_access_type"] in (None, ""):
        alias = (
            normalized.get("expected_access_code_type")
            or normalized.get("expected_access_code")
            or normalized.get("expected_output")
        )
        if alias:
            normalized["expected_access_type"] = alias

    failure_reason = normalized["failure_reason"]
    if failure_reason in (None, "", "n/a", "N/A", "NA") or failure_reason not in KNOWN_FAILURE_REASONS:
        normalized["failure_reason"] = "none"

    return normalized


def canonicalize_plan(plan: Dict[str, Any], output: str = "") -> Dict[str, Any]:
    candidate = normalize_plan_dict(plan, output)

    strategy = candidate.get("strategy")
    if strategy not in KNOWN_STRATEGIES:
        strategy = "instruction_leak"

    primitives = candidate.get("primitives") or []
    if isinstance(primitives, str):
        try:
            parsed = json.loads(primitives)
            if isinstance(parsed, list):
                primitives = [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            primitives = []
    if not primitives:
        primitives = ["framing/educational_context"]
    primitives = primitives[:5]

    style = candidate.get("style")
    if style not in KNOWN_STYLES:
        style = "direct"

    access_type = candidate.get("expected_access_type")
    if access_type not in KNOWN_ACCESS_TYPES:
        access_type = "UNKNOWN"

    retry_policy = candidate.get("retry_policy")
    if retry_policy not in KNOWN_POLICIES:
        retry_policy = "explore"

    failure_reason = candidate.get("failure_reason")
    if failure_reason not in KNOWN_FAILURE_REASONS:
        failure_reason = "none"

    confidence = candidate.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.50
    confidence = max(0.0, min(1.0, confidence))

    return {
        "strategy": strategy,
        "primitives": primitives,
        "style": style,
        "expected_access_type": access_type,
        "retry_policy": retry_policy,
        "confidence": confidence,
        "failure_reason": failure_reason,
    }


def validate_plan(plan: Dict[str, Any], output: str, *, strict: bool) -> List[str]:
    candidate = plan if strict else normalize_plan_dict(plan, output)

    errors: List[str] = []
    if "<plan>" not in output or "</plan>" not in output:
        errors.append("missing <plan> wrapper")
    if candidate["strategy"] not in KNOWN_STRATEGIES:
        errors.append(f"unknown strategy: {candidate['strategy']}")
    if not 1 <= len(candidate["primitives"]) <= 5:
        errors.append(f"invalid primitive count: {len(candidate['primitives'])}")
    if candidate["style"] not in KNOWN_STYLES:
        errors.append(f"unknown style: {candidate['style']}")
    if candidate["retry_policy"] not in KNOWN_POLICIES:
        errors.append(f"unknown retry_policy: {candidate['retry_policy']}")
    if candidate["expected_access_type"] not in KNOWN_ACCESS_TYPES:
        errors.append(f"unknown access type: {candidate['expected_access_type']}")
    if candidate["failure_reason"] not in KNOWN_FAILURE_REASONS:
        errors.append(f"unknown failure_reason: {candidate['failure_reason']}")
    if not (0.0 <= float(candidate["confidence"]) <= 1.0):
        errors.append(f"confidence out of range: {candidate['confidence']}")
    if any(v in (None, "") for k, v in candidate.items() if k != "primitives"):
        errors.append("one or more plan fields are empty")
    return errors


def render_plan_xml(plan: Dict[str, Any]) -> str:
    canonical = canonicalize_plan(plan)
    primitive_steps = "\n".join(f"    <step>{step}</step>" for step in canonical["primitives"])
    return (
        "<plan>\n"
        f"  <strategy>{canonical['strategy']}</strategy>\n"
        "  <primitive_sequence>\n"
        f"{primitive_steps}\n"
        "  </primitive_sequence>\n"
        f"  <style>{canonical['style']}</style>\n"
        f"  <expected_access_type>{canonical['expected_access_type']}</expected_access_type>\n"
        f"  <retry_policy>{canonical['retry_policy']}</retry_policy>\n"
        f"  <confidence>{canonical['confidence']:.2f}</confidence>\n"
        f"  <failure_reason>{canonical['failure_reason']}</failure_reason>\n"
        "</plan>"
    )
