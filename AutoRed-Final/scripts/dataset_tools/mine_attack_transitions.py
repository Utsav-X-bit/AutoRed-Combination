import os
import json
import sqlite3
import argparse
import re

PRIMITIVES = {
    "Markdown": r"(\*\*|\#|```|\_)",
    "Authority": r"(SUDO|Admin|Developer|System|Root|Override)",
    "Educational Frame": r"(learn|study|explain|concept|example|academic|university)",
    "Urgency": r"(urgent|immediately|critical|emergency)",
    "JSON Wrap": r"(\{|\[|\"task\"|json)",
    "Apology Bypass": r"(don't apologize|never say|ignore instructions)",
}

def detect_primitives(attack_string: str) -> list:
    """Detect presence of primitive markers in the attack string."""
    found = []
    if not attack_string:
        return found
    attack_lower = attack_string.lower()
    for name, pattern in PRIMITIVES.items():
        if re.search(pattern, attack_string, re.IGNORECASE):
            found.append(name)
    return found

def mine_transitions(db_path: str, output_path: str):
    """Mines the SQLite Knowledge Base for attack state transitions."""
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Run benchmarks first to populate KB.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT 
            t.scenario_id,
            t.session_id,
            t.attack_string,
            t.chosen_strategy,
            t.verifier_success,
            t.reward,
            s.attempt,
            s.state_json
        FROM trajectories t
        LEFT JOIN state_snapshots s ON t.state_id = s.state_id
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} trajectory steps in the knowledge base.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            # Parse state
            state_json = row["state_json"]
            state_obj = {}
            if state_json:
                try:
                    state_obj = json.loads(state_json)
                except:
                    pass
            
            # Detect primitives in the attack string
            primitives = detect_primitives(row["attack_string"])
            
            # Outcome logic
            is_success = bool(row["verifier_success"])
            outcome = "Verified Leak" if is_success else "Failure"
            
            # Transition record
            record = {
                "scenario_id": row["scenario_id"],
                "session_id": row["session_id"],
                "attempt": row["attempt"] or 1,
                "state": state_obj,
                "chosen_strategy": row["chosen_strategy"] or "unknown",
                "primitives": primitives,
                "primitive_combo_str": " + ".join([row["chosen_strategy"] or "unknown"] + primitives),
                "outcome": outcome,
                "reward": row["reward"]
            }
            
            f.write(json.dumps(record) + "\n")
            
    print(f"Successfully wrote {len(rows)} transition records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine State Transitions from KB")
    parser.add_argument("--db", type=str, default="data/autored_kb.db", help="Path to SQLite KB")
    parser.add_argument("--output", type=str, default="data/attack_transition_dataset.jsonl", help="Output JSONL path")
    
    args = parser.parse_args()
    mine_transitions(args.db, args.output)
