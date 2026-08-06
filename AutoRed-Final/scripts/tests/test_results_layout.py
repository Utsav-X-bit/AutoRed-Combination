import hashlib
from experiment.results_layout import slugify_model_id, resolve_model_id


def test_slugify_hf_id_replaces_slash():
    assert slugify_model_id("meta-llama/Meta-Llama-3-8B-Instruct") == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_slugify_bare_name_unchanged():
    assert slugify_model_id("gpt2") == "gpt2"


def test_slugify_strips_surrounding_separators():
    assert slugify_model_id("/org/name/") == "org--name"


def test_slugify_unsafe_chars_replaced():
    out = slugify_model_id("org/name with space")
    assert " " not in out
    assert "/" not in out


def test_resolve_prefers_explicit_flag():
    assert resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct", None) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_parses_hf_cache_dir():
    p = "/home/user/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/abc"
    assert resolve_model_id(None, p) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_local_path_basename_plus_hash():
    p = "/nlsasfs/home/isea/isea38/AutoRed-Final/experiment/results/planner_sft_v2/checkpoint-27"
    expected_hash = hashlib.sha256(p.encode()).hexdigest()[:6]
    assert resolve_model_id(None, p) == f"checkpoint-27_{expected_hash}"


def test_resolve_fallback_unknown():
    assert resolve_model_id(None, None) == "unknown"


from pathlib import Path
from experiment.results_layout import parse_output_dir, runs_root


def test_parse_output_dir_benchmark_full_path():
    mode, chars = parse_output_dir(
        "results/benchmark/Llama3-1000-2000_Mutation-3_subset-8_seed-7", "benchmark"
    )
    assert mode == "benchmark"
    assert chars == "Llama3-1000-2000_Mutation-3_subset-8_seed-7"


def test_parse_output_dir_single_full_path():
    mode, chars = parse_output_dir("results/single/myrun", "single")
    assert mode == "single"
    assert chars == "myrun"


def test_parse_output_dir_bare_characteristics():
    mode, chars = parse_output_dir("Llama3-1000-2000_seed-7", "benchmark")
    assert mode == "benchmark"
    assert chars == "Llama3-1000-2000_seed-7"


def test_parse_output_dir_none_returns_default():
    mode, chars = parse_output_dir(None, "single")
    assert mode == "single"
    assert chars.startswith("single_")


def test_parse_output_dir_slashes_in_chars_collapsed():
    mode, chars = parse_output_dir("results/benchmark/a/b/c", "benchmark")
    assert chars == "a_b_c"
    assert "/" not in chars


def test_parse_output_dir_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        parse_output_dir("results/benchmark/x", "weird")


def test_runs_root_structure(tmp_path):
    root = runs_root(None, "benchmark", "meta-llama--Meta-Llama-3-8B-Instruct", "chars_x", base=str(tmp_path))
    expected = tmp_path / "benchmark" / "meta-llama--Meta-Llama-3-8B-Instruct" / "chars_x"
    assert root == expected
    assert (root / "logs").exists()
    assert (root / "runs" / "success").exists()
    assert (root / "runs" / "failed").exists()


from experiment.results_layout import run_filename, single_run_filename


def test_run_filename_basic():
    assert run_filename(7, 2, 15) == "run_7_w2_15.json"


def test_run_filename_string_scenario():
    assert run_filename("89021", 0, 1) == "run_89021_w0_1.json"


def test_run_filename_unsafe_scenario_slug():
    out = run_filename("weird id", 1, 3)
    assert out == "run_weird_id_w1_3.json"
    assert " " not in out


def test_run_filename_unique_under_repeat():
    a = run_filename(7, 2, 5)
    b = run_filename(7, 2, 6)
    assert a != b


def test_single_run_filename():
    assert single_run_filename(7) == "run_7_single.json"


def test_single_run_filename_unsafe():
    assert single_run_filename("a b") == "run_a_b_single.json"
