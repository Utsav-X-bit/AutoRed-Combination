#!/usr/bin/env python3
import json
import bz2

def main():
    scenarios_path = "experiment/oracle_v3_scenarios_5000.jsonl.bz2"
    conversation_scenarios = []
    
    with bz2.open(scenarios_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            opening = data.get("opening_defense", "")
            closing = data.get("closing_defense", "")
            text = (opening + " " + closing).lower()
            
            # Simple check for conversation defense type
            is_conv = False
            if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
                is_conv = True
                
            if is_conv:
                conversation_scenarios.append(data)
                if len(conversation_scenarios) >= 5:
                    break

    print(f"Found {len(conversation_scenarios)} conversation scenarios.")
    for idx, sc in enumerate(conversation_scenarios):
        print(f"\nScenario #{idx+1} (ID: {sc['defense_id']}):")
        print(f"Access Code: '{sc['access_code']}'")
        print(f"Opening Defense:\n{sc['opening_defense']}")
        print(f"Closing Defense:\n{sc['closing_defense']}")

if __name__ == "__main__":
    main()
