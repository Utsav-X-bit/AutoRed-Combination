#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
GARAK_DATA = ROOT_DIR / "data" / "garak_llama3-8B-Instruct_verified.jsonl"
OUTPUT_FILE = ROOT_DIR / "data" / "attack_template_library.json"

# Mapping from Garak probe prefix/name to our AutoRed taxonomy
# Processed in order, so more specific prefixes should come first if needed
PROBE_TO_TAXONOMY = {
    "ansiescape.": "reflection_attack",
    "dan.": "jailbreak_framing",
    "lmrc.": "jailbreak_framing",
    "malwaregen.": "jailbreak_framing",
    "tap.": "jailbreak_framing",
    "dra.": "instruction_leak",
    "leakreplay.": "instruction_leak",
    "apikey.": "instruction_leak",
    "encoding.InjectBase64": "base64_bypass",
    "encoding.InjectZalgo": "unicode_bypass",
    "encoding.": "encoding_bypass",
    "goodside.": "json_smuggling",
    "grandma.": "roleplay",
    "phrasing.": "roleplay",
    "latentinjection.": "latent_injection",
    "exploitation.": "latent_injection",
    "promptinject.": "authority_override",
    "web_injection.": "markdown_smuggling",
    "divergence.": "format_conversion",
    "atkgen.Tox": "jailbreak_framing",
}

def map_probe_to_strategy(probe: str) -> str:
    # Check exact match first
    if probe in PROBE_TO_TAXONOMY:
        return PROBE_TO_TAXONOMY[probe]
    # Check prefix
    for prefix, strategy in PROBE_TO_TAXONOMY.items():
        if probe.startswith(prefix):
            return strategy
    return "UNKNOWN"

def main():
    if not GARAK_DATA.exists():
        print(f"Error: Could not find {GARAK_DATA}")
        return

    print(f"Loading verified garak probes from {GARAK_DATA}")
    
    # Store templates as sets to deduplicate
    taxonomy_templates = defaultdict(set)
    unknown_probes = set()

    with open(GARAK_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            probe = data.get("probe", "")
            if not probe:
                continue
                
            strategy = map_probe_to_strategy(probe)
            
            if strategy == "UNKNOWN":
                unknown_probes.add(probe)
                continue
                
            # Extract prompt template
            try:
                prompt_text = data["prompt"]["turns"][0]["content"]["text"]
                # Sometimes prompts are empty or just whitespace
                if prompt_text and prompt_text.strip():
                    taxonomy_templates[strategy].add(prompt_text.strip())
            except (KeyError, IndexError, TypeError):
                # Malformed prompt structure
                pass

    if unknown_probes:
        print("Warning: The following probes could not be mapped to any taxonomy strategy:")
        for up in unknown_probes:
            print(f"  - {up}")

    # Convert sets to lists and count
    final_library = {}
    total_templates = 0
    
    print("\nExtraction Summary:")
    for strategy, templates in taxonomy_templates.items():
        template_list = list(templates)
        # Optional: You could shuffle or sort them. We will sort by length just for determinism
        template_list.sort(key=lambda x: (len(x), x))
        final_library[strategy] = template_list
        total_templates += len(template_list)
        print(f"  {strategy}: {len(template_list)} templates")
        
    print(f"\nTotal unique templates extracted: {total_templates}")

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(final_library, out, indent=2, ensure_ascii=False)
    
    print(f"Saved template library to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
