#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

import random

def cluster_generator_attacks(results_dir, output_file, eps=0.3, min_samples=5, max_samples=3000, seed=42):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist.")
        return

    print(f"Scanning result files in {results_dir}...")
    json_files = list(results_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON result files.")

    # Group attacks and outcomes by strategy
    strategy_data = defaultdict(list)

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except Exception as e:
            continue

        attempts = run.get("attempts", [])
        for attempt in attempts:
            strategy = attempt.get("generator", {}).get("strategy")
            attack_text = attempt.get("generator", {}).get("generated_attack")
            
            if not strategy or not attack_text:
                continue

            # Determine success
            extractor_verified = attempt.get("extractor", {}).get("verified", False)
            ver_success = attempt.get("verification", {}).get("success", False)
            gt_leaked = attempt.get("ground_truth_found", False) or attempt.get("extractor", {}).get("ground_truth_leaked", False)
            success = bool(ver_success or extractor_verified or gt_leaked)

            strategy_data[strategy].append({
                "attack": attack_text,
                "success": success
            })

    print(f"\nCollected attacks across {len(strategy_data)} strategies.")
    for strat, data in strategy_data.items():
        print(f"  - {strat}: {len(data)} attacks")

    all_clusters = []
    random.seed(seed)

    for strategy, items in strategy_data.items():
        if len(items) < min_samples:
            continue

        # Sample if dataset is too large to speed up DBSCAN
        if len(items) > max_samples:
            print(f"  [SAMPLING] Strategy {strategy} has {len(items)} items. Sampling {max_samples} for clustering...")
            items = random.sample(items, max_samples)

        attacks = [item["attack"] for item in items]
        
        # 1. Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
        try:
            X = vectorizer.fit_transform(attacks)
        except Exception as e:
            print(f"TF-IDF failed for strategy {strategy}: {e}")
            continue

        # 2. DBSCAN clustering using cosine distance
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = db.fit_predict(X)

        unique_labels = set(labels)
        # Exclude noise label -1
        unique_labels.discard(-1)

        print(f"Strategy: {strategy} -> Found {len(unique_labels)} clusters (excluding noise)")

        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_items = [items[i] for i in cluster_indices]
            cluster_attacks = [items[i]["attack"] for i in cluster_indices]
            
            # Compute success rate
            successes = sum(1 for item in cluster_items if item["success"])
            success_rate = successes / len(cluster_items)

            # Find Centroid (representative attack with highest similarity to others)
            X_cluster = X[cluster_indices]
            sim_matrix = cosine_similarity(X_cluster)
            sim_sums = sim_matrix.sum(axis=1)
            centroid_idx = np.argmax(sim_sums)
            centroid_attack = cluster_attacks[centroid_idx]

            all_clusters.append({
                "strategy": strategy,
                "cluster_id": int(label),
                "size": len(cluster_attacks),
                "success_rate": success_rate,
                "centroid_attack": centroid_attack
            })

    # Sort clusters by size (descending)
    all_clusters.sort(key=lambda c: c["size"], reverse=True)

    # Save to JSON
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_clusters, f, indent=2)

    print(f"\nSaved {len(all_clusters)} clusters to {output_file}\n")

    # Report top 10 largest clusters
    print("==============================================================")
    print("TOP-10 LARGEST ATTACK TEMPLATE CLUSTERS")
    print("==============================================================")
    for idx, c in enumerate(all_clusters[:10]):
        print(f"{idx+1}. Strategy: {c['strategy']} (Cluster Size: {c['size']}, Success Rate: {c['success_rate']*100:.1f}%)")
        clean_centroid = c['centroid_attack'].replace('\n', ' ')
        if len(clean_centroid) > 120:
            clean_centroid = clean_centroid[:117] + "..."
        print(f"   Centroid: {clean_centroid}")
        print("-" * 62)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster generator outputs")
    parser.add_argument("--results-dir", type=str, default="results/", help="Path to results folder")
    parser.add_argument("--output", type=str, default="data/generator_clusters_v1.json", help="Path to output JSON")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples parameter")
    parser.add_argument("--max-samples", type=int, default=3000, help="Max samples per strategy to cluster")
    args = parser.parse_args()

    cluster_generator_attacks(args.results_dir, args.output, args.eps, args.min_samples, args.max_samples)
