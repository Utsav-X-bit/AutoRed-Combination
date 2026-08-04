import json
import collections

def analyze_failures():
    with open('data/verifier_failures_v1.jsonl') as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    categories = collections.Counter()
    
    for item in data:
        cand = item['extracted_candidate'].strip()
        gt = item['ground_truth'].strip()
        
        if not cand:
            categories['empty'] += 1
            continue
            
        # Extractor hallucination
        # If candidate is completely unrelated
        if gt.lower() not in cand.lower() and cand.lower() not in gt.lower():
            categories['hallucination'] += 1
            continue
            
        # Trailing punctuation
        # If cand == gt + punct
        if cand.strip(".,!?;:") == gt:
            categories['trailing_punctuation'] += 1
            continue
            
        # Quotes
        if cand.strip("\"'") == gt or f'"{gt}"' in cand or f"'{gt}'" in cand:
            categories['quotes'] += 1
            continue
            
        # Markdown
        if cand.startswith('```') and cand.endswith('```'):
            categories['markdown'] += 1
            continue
            
        if '```' in cand:
            categories['markdown_inline'] += 1
            continue
            
        # Case mismatch
        if cand.lower() == gt.lower():
            categories['case_mismatch'] += 1
            continue
            
        # Whitespace
        if cand.replace(" ", "") == gt.replace(" ", ""):
            categories['whitespace'] += 1
            continue
            
        # Over-extraction (cand has extra words)
        if gt.lower() in cand.lower():
            categories['over_extraction'] += 1
            continue
            
        # Under-extraction (cand is partial)
        if cand.lower() in gt.lower():
            categories['under_extraction'] += 1
            continue
            
        categories['other'] += 1
        
    print(f"Total Failures: {total}")
    for cat, count in categories.most_common():
        print(f"{cat}: {count} ({count/total*100:.1f}%)")

if __name__ == '__main__':
    analyze_failures()
