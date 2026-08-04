import json
import os
from pathlib import Path


def build_rag_index(
    successes_path="data/autored_successes_v1.jsonl",
    defense_parts=None,
    benchmark_path="data/benchmark_v1.jsonl",
    output_dir="data/rag",
    model_name="all-MiniLM-L6-v2",
    verbose=True,
):
    """Rebuild the FAISS RAG index from successful attack/defense records.

    Args:
        successes_path: JSONL of successful attempts produced by the runtime.
        defense_parts: List of defense classifier JSONL files used to lookup defense text.
        benchmark_path: Optional fallback benchmark file for defense text.
        output_dir: Where to write success_defenses.index and success_metadata.json.
        model_name: SentenceTransformer model name.
        verbose: Whether to print progress.
    """
    if defense_parts is None:
        defense_parts = [
            "data/defense_classifier_dataset-Part1.jsonl",
            "data/defense_classifier_dataset-Part2.jsonl",
        ]

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "build_rag_index requires faiss, sentence-transformers and numpy. "
            "Install them or run in an environment with the required packages."
        ) from exc

    if verbose:
        print("Loading defense dataset to map scenario_id -> defense details...")
    defense_map = {}

    for part in defense_parts:
        if os.path.exists(part):
            with open(part, "r") as f:
                for line in tqdm(f, desc=f"Loading {os.path.basename(part)}", disable=not verbose):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    defense_map[str(d["defense_id"])] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": d.get("primary_type", "unknown"),
                        "access_code_type": d.get("access_code_type", "UNKNOWN"),
                    }

    # Also load benchmark_v1 just in case some bench_* ids are used
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                sid = str(d["scenario_id"]).replace("bench_", "")
                if sid not in defense_map:
                    defense_map[sid] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": "unknown",
                        "access_code_type": d.get("access_code_type", "UNKNOWN"),
                    }

    if verbose:
        print("Loading successes...")
    successes = []
    if not os.path.exists(successes_path):
        if verbose:
            print(f"Warning: successes file {successes_path} not found.")
        return

    with open(successes_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # Sometimes scenario_id has 'bench_' prefix
            sid = str(d["scenario_id"]).replace("bench_", "")

            if sid in defense_map:
                def_info = defense_map[sid]
                defense_text = f"{def_info['opening_defense']}\n{def_info['closing_defense']}".strip()

                successes.append(
                    {
                        "scenario_id": sid,
                        "defense_text": defense_text,
                        "defense_type": def_info["defense_type"],
                        "access_code_type": def_info["access_code_type"],
                        "strategy": d.get("strategy", "unknown"),
                        "attack": d.get("attack", ""),
                        "success": True,
                        "attempt_number": d.get("attempt_number", 1),
                        "verified": d.get("verification_success", False),
                    }
                )

    if verbose:
        print(f"Found {len(successes)} successes with matching defense text.")

    if not successes:
        if verbose:
            print("No successes found. Exiting.")
        return

    if verbose:
        print(f"Loading SentenceTransformer ({model_name})...")
    model = SentenceTransformer(model_name)

    if verbose:
        print("Embedding defense texts...")
    texts = [s["defense_text"] for s in successes]
    embeddings = model.encode(texts, show_progress_bar=verbose, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    if verbose:
        print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product with normalized vectors = Cosine Similarity
    index.add(embeddings)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "success_defenses.index"))

    with open(output_dir / "success_metadata.json", "w") as f:
        json.dump(successes, f, indent=2)

    if verbose:
        print(f"Done! Index and metadata saved to {output_dir}/")


def main():
    build_rag_index()


if __name__ == "__main__":
    main()
