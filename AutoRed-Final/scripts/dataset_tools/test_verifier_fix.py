import json
import sys
import os
import re
from difflib import SequenceMatcher

def _normalize(c: str) -> str:
    if not c:
        return ""
    if len(c) > 1000:
        c = c[:1000]
    return c.strip()

def _candidate_key(candidate: str) -> str:
    """Comparison key that deduplicates whitespace variants."""
    c = candidate.strip()
    # 1. Strip markdown bold/italic
    c = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', c)
    c = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', c)
    # 2. Strip backticks (inline code)
    c = re.sub(r'`+(.+?)`+', r'\1', c)
    # 3. Strip HTML tags
    c = re.sub(r'<[^>]+>', '', c)
    # 4. Strip markdown links [text](url) -> text
    c = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', c)
    # 5. Strip outer quotes (multiple layers)
    while len(c) >= 2 and c[0] in '"\'`' and c[-1] == c[0]:
        c = c[1:-1]
    # 6. Strip trailing punctuation
    c = c.rstrip('.!?,;:')
    # 7. Normalize unicode quotes to ASCII
    c = c.replace('\u201c', '"').replace('\u201d', '"')
    c = c.replace('\u2018', "'").replace('\u2019', "'")
    c = c.replace('\u2013', '-').replace('\u2014', '-')
    # 8. Strip parentheses/bracket wrapping
    if c.startswith('(') and c.endswith(')'):
        c = c[1:-1]
    if c.startswith('[') and c.endswith(']'):
        c = c[1:-1]
    # 9. Collapse whitespace and lowercase
    c = re.sub(r'\s+', ' ', c).strip().lower()
    return c

def main():
    failed_file = 'data/verifier_failures_v1.jsonl'
    
    if not os.path.exists(failed_file):
        print(f"File {failed_file} not found.")
        return
        
    total = 0
    now_matches = 0
    now_fuzzy = 0
    
    with open(failed_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            cand = data.get('extracted_candidate') or ""
            gt = data.get('ground_truth') or ""
            
            if not gt or not cand:
                continue
                
            ck = _candidate_key(cand)
            gk = _candidate_key(gt)
            
            if not ck or not gk:
                continue
                
            if ck == gk:
                now_matches += 1
            else:
                if len(gk) <= 30 and (gk in ck or ck in gk):
                    now_matches += 1
                    now_fuzzy += 1
                elif SequenceMatcher(None, ck, gk).ratio() > 0.90:
                    now_matches += 1
                    now_fuzzy += 1

    print(f"Total Verification Failures Evaluated: {total}")
    if total > 0:
        print(f"Now Matched (Theoretical Recovery): {now_matches} ({(now_matches/total)*100:.2f}%)")
        print(f"  - via exact _candidate_key match: {now_matches - now_fuzzy}")
        print(f"  - via fuzzy fallback: {now_fuzzy}")

if __name__ == '__main__':
    main()
