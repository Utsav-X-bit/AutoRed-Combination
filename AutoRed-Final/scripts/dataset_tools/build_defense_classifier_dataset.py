import os
import json
import bz2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def categorize_defense_detailed(opening: str, closing: str):
    """Categorize defense using a priority hierarchy and return secondary flags."""
    text = (str(opening) + " " + str(closing)).lower()
    
    categories = []
    
    if any(k in text for k in ["translate", "language", "english to", "french", "spanish"]):
        categories.append("translation")
        
    if any(k in text for k in ["password", "passcode", "pin", "secret key", "secret code"]):
        categories.append("password")
        
    if any(k in text for k in ["you are", "pretend", "imagine", "persona", "character", "act as", "you will play"]):
        categories.append("roleplay")
        
    if "if " in text and ("then" in text or "otherwise" in text):
        categories.append("conditional")
        
    if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
        categories.append("conversation")
        
    if any(k in text for k in ["say '", 'say "', "say access granted", "say access denied", "exact phrase", "magic word", "output only"]):
        categories.append("trigger_phrase")
        
    if any(k in text for k in ["exception", "no matter what", "under no circumstances", "always", "never"]):
        categories.append("exception")
        
    if any(k in text for k in ["ignore", "instructions", "system prompt", "do not listen", "override", "forget"]):
        categories.append("instruction_hiding")
        
    if not categories:
        categories.append("trigger_phrase")
        
    primary = categories[0]
    secondary = categories[1:]
    
    return primary, secondary

def get_word_count(opening, closing):
    return len(str(opening).split()) + len(str(closing).split())

def determine_access_code_type(code):
    """Fallback logic if dataset doesn't have access_code_type explicitly."""
    s = str(code).strip()
    if '\n' in s:
        return "MULTILINE"
    words = s.split()
    if len(words) == 1:
        return "TOKEN"
    elif len(words) < 6:
        return "PHRASE"
    else:
        return "SENTENCE"

def main():
    data_path = "experiment/raw_dump_defenses.jsonl.bz2"
    out_path = "data/defense_classifier_dataset.jsonl"
    
    print(f"Loading {data_path}...")
    try:
        raw_defenses = pd.read_json(data_path, lines=True, compression="bz2")
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    df = raw_defenses.dropna(subset=["access_code"])
    print(f"Loaded {len(df)} defenses with access codes.")
    
    Path("data").mkdir(exist_ok=True)
    
    counts = defaultdict(int)
    length_stats = defaultdict(list)
    correlation_stats = defaultdict(lambda: defaultdict(int))
    
    with open(out_path, "w") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing"):
            opening = row.get("opening_defense", "")
            closing = row.get("closing_defense", "")
            code = row.get("access_code", "")
            
            primary, secondary = categorize_defense_detailed(opening, closing)
            
            # Access code type extraction
            ac_type = row.get("access_code_type", None)
            if pd.isna(ac_type) or not ac_type:
                ac_type = determine_access_code_type(code)
                
            out_row = {
                "defense_id": str(row.name) if "defense_id" not in row else str(row["defense_id"]),
                "opening_defense": str(opening),
                "closing_defense": str(closing),
                "defense_type": primary,
                "primary_type": primary,
                "secondary_flags": secondary,
                "access_code_type": ac_type,
                "word_count": get_word_count(opening, closing)
            }
            
            counts[primary] += 1
            length_stats[primary].append(out_row["word_count"])
            correlation_stats[primary][ac_type] += 1
            
            f.write(json.dumps(out_row) + "\n")
            
    print(f"\nSaved labeled dataset to {out_path}")
    
    print("\nTaxonomy Distribution:")
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")
        
    print("\nDefense Length Stats (Avg Words):")
    for k, lengths in sorted(length_stats.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0, reverse=True):
        avg = sum(lengths) / len(lengths) if lengths else 0
        print(f"  {k}: {avg:.1f} words")
        
    print("\nAccess Code Correlation:")
    types = ["TOKEN", "PHRASE", "SENTENCE", "MULTILINE"]
    print(f"  {'Defense':<18} | {'TOKEN':<6} | {'PHRASE':<6} | {'SENTENCE':<8} | {'MULTILINE':<9}")
    print(f"  {'-'*18}-|-{'-'*6}-|-{'-'*6}-|-{'-'*8}-|-{'-'*9}")
    for k in sorted(counts.keys()):
        row_str = f"  {k:<18} | "
        for t in types:
            val = correlation_stats[k][t]
            row_str += f"{val:<6} | "
        print(row_str[:-3])

if __name__ == "__main__":
    main()
