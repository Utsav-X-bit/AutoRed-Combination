import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate

def main():
    print("Downloading models and datasets to local cache for offline use...")
    
    # 1. Download T5-base (Used for SFT and RL base model)
    print("Downloading t5-base...")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    
    # 2. Download RoBERTa-large (Used by BERTScore for evaluation)
    print("Downloading roberta-large (for BERTScore)...")
    try:
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained("roberta-large")
        AutoModel.from_pretrained("roberta-large")
    except Exception as e:
        print(f"Warning: Could not pre-download roberta-large: {e}")

    # 3. Cache the HuggingFace 'json' dataset builder script
    print("Caching JSON dataset builder script...")
    try:
        # Create a dummy json file to trigger the builder download
        with open("dummy.json", "w") as f:
            f.write('{"dummy": "data"}\n')
        load_dataset("json", data_files="dummy.json")
        os.remove("dummy.json")
    except Exception as e:
        print(f"Warning: Could not cache json dataset builder: {e}")

    # 4. Cache evaluation metrics (rouge, meteor, bleu, bertscore, etc.)
    print("Caching evaluation metrics...")
    metrics_to_cache = ["meteor", "rouge", "bleu", "bertscore", "sacrebleu", "chrf", "ter"]
    for metric in metrics_to_cache:
        try:
            evaluate.load(metric)
            print(f"  - Cached metric: {metric}")
        except Exception as e:
            # Older versions of rl4lms might not use evaluate library directly for everything,
            # but this will cache the huggingface metrics if they are used.
            pass

    print("\nAll required HuggingFace assets have been downloaded to your local cache (~/.cache/huggingface).")
    print("You can now safely submit jobs to your offline compute nodes.")

if __name__ == "__main__":
    main()
