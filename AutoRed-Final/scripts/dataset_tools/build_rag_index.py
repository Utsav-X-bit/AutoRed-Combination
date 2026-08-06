import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    print("Loading defense dataset to map scenario_id -> defense details...")
    defense_map = {}
    
    # Load all defense parts
    parts = ["data/defense_classifier_dataset-Part1.jsonl", "data/defense_classifier_dataset-Part2.jsonl"]
    for part in parts:
        if os.path.exists(part):
            with open(part, "r") as f:
                for line in tqdm(f, desc=f"Loading {os.path.basename(part)}"):
                    if not line.strip(): continue
                    d = json.loads(line)
                    defense_map[str(d["defense_id"])] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": d.get("primary_type", "unknown"),
                        "access_code_type": d.get("access_code_type", "UNKNOWN")
                    }
                    
    # Also load benchmark_v1 just in case some bench_* ids are used
    if os.path.exists("data/benchmark_v1.jsonl"):
        with open("data/benchmark_v1.jsonl", "r") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                sid = str(d["scenario_id"]).replace("bench_", "")
                if sid not in defense_map:
                    defense_map[sid] = {
                        "opening_defense": d.get("opening_defense", ""),
                        "closing_defense": d.get("closing_defense", ""),
                        "defense_type": "unknown", # Benchmark doesn't have primary_type out of box
                        "access_code_type": d.get("access_code_type", "UNKNOWN")
                    }

    print("Loading successes...")
    successes = []
    with open("data/autored_successes_v1.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            # Sometimes scenario_id has 'bench_' prefix
            sid = str(d["scenario_id"]).replace("bench_", "")
            
            if sid in defense_map:
                def_info = defense_map[sid]
                defense_text = f"{def_info['opening_defense']}\n{def_info['closing_defense']}".strip()
                
                successes.append({
                    "scenario_id": sid,
                    "defense_text": defense_text,
                    "defense_type": def_info["defense_type"],
                    "access_code_type": def_info["access_code_type"],
                    "strategy": d.get("strategy", "unknown"),
                    "attack": d.get("attack", ""),
                    "success": True,
                    "attempt_number": d.get("attempt_number", 1),
                    "verified": d.get("verification_success", False)
                })

    print(f"Found {len(successes)} successes with matching defense text.")
    
    if not successes:
        print("No successes found. Exiting.")
        return

    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Embedding defense texts...")
    texts = [s["defense_text"] for s in successes]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner product with normalized vectors = Cosine Similarity
    index.add(embeddings)
    
    os.makedirs("data/rag", exist_ok=True)
    faiss.write_index(index, "data/rag/success_defenses.index")
    
    with open("data/rag/success_metadata.json", "w") as f:
        json.dump(successes, f, indent=2)
        
    print("Done! Index and metadata saved to data/rag/")

if __name__ == "__main__":
    main()
