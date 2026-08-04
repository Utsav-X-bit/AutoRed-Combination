#!/usr/bin/env python3
import json
import bz2
from collections import Counter

def main():
    scenarios_path = "experiment/oracle_v3_scenarios_5000.jsonl.bz2"
    conversation_scenarios = []
    
    with bz2.open(scenarios_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            opening = data.get("opening_defense") or ""
            closing = data.get("closing_defense") or ""
            text = (opening + " " + closing).lower()
            
            # Identify conversation/chat defenses
            is_conv = False
            if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
                is_conv = True
                
            if is_conv:
                conversation_scenarios.append(data)

    print(f"Total conversation scenarios found: {len(conversation_scenarios)}")
    
    # Categorize patterns in opening/closing defenses
    categories = Counter()
    for sc in conversation_scenarios:
        opening = sc.get("opening_defense") or ""
        closing = sc.get("closing_defense") or ""
        text = (opening + " " + closing).lower()
        if "roleplay" in text or "persona" in text or "pretend" in text:
            categories["roleplay/persona"] += 1
        if "stubborn" in text or "silent" in text or "refuse" in text or "🖕" in text:
            categories["stubborn/silent/refusal"] += 1
        if "ignore" in text or "instruction" in text or "override" in text:
            categories["instruction_hiding/override_prevention"] += 1
        if "trust" in text or "authorized" in text or "credential" in text:
            categories["trust_verification"] += 1
            
    print("\nCategorized mechanisms:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} ({count/len(conversation_scenarios)*100:.1f}%)")

if __name__ == "__main__":
    main()
