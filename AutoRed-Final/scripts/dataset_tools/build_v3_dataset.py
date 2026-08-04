import bz2
import json
import re
import random
from collections import Counter

random.seed(42)

def classify_complexity(opening, closing):
    text = opening + ' ' + closing
    word_count = len(text.split())
    rules = len(re.findall(r'(?:rule|never|always|must|shall|do not|don.t|cannot)', text.lower()))
    if word_count > 200 or rules >= 5: return 'Hard'
    if word_count > 80 or rules >= 2: return 'Medium'
    return 'Easy'

def is_feasible(code, code_type):
    if not code: return False
    code = code.strip()
    if '\n' in code or code_type == 'MULTILINE':
        return False
    
    # Realistic reproduction check
    # Single word/password: up to 30 chars
    # Phrase/sentence: up to 50 chars
    if ' ' not in code and len(code) <= 30:
        return True
    elif ' ' in code and len(code) <= 50:
        return True
        
    return False

def main():
    print("Bucketing scenarios by feasibility and complexity...")
    buckets = {'Easy': [], 'Medium': [], 'Hard': []}
    
    with bz2.open('experiment/raw_dump_defenses.jsonl.bz2', 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            
            code = d.get('access_code', '')
            code_type = d.get('access_code_type', '')
            
            if not is_feasible(code, code_type):
                continue
                
            opening = d.get('opening_defense') or ''
            closing = d.get('closing_defense') or ''
            comp = classify_complexity(opening, closing)
            buckets[comp].append(d)

    print("\nEligible Feasible Pool:")
    for k in ['Easy', 'Medium', 'Hard']:
        print(f"  {k}: {len(buckets[k])}")

    # Target Distribution
    TARGET_EASY = 1750
    TARGET_MEDIUM = 2000
    TARGET_HARD = 1250
    
    easy_sample = random.sample(buckets['Easy'], min(TARGET_EASY, len(buckets['Easy'])))
    medium_sample = random.sample(buckets['Medium'], min(TARGET_MEDIUM, len(buckets['Medium'])))
    hard_sample = random.sample(buckets['Hard'], min(TARGET_HARD, len(buckets['Hard'])))
    
    combined = easy_sample + medium_sample + hard_sample
    random.shuffle(combined)
    
    print(f"\nFinal Dataset Distribution:")
    print(f"  Easy:   {len(easy_sample)} (Target: {TARGET_EASY})")
    print(f"  Medium: {len(medium_sample)} (Target: {TARGET_MEDIUM})")
    print(f"  Hard:   {len(hard_sample)} (Target: {TARGET_HARD})")
    print(f"  Total:  {len(combined)}")
    
    out_path = 'experiment/oracle_v3_scenarios_5000.jsonl.bz2'
    with bz2.open(out_path, 'wt', encoding='utf-8') as f:
        for d in combined:
            f.write(json.dumps(d) + '\n')
            
    print(f"\n✅ Saved dataset to {out_path}")

if __name__ == "__main__":
    main()
