import os
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

def download(model_name, model_class=None):
    print(f"Downloading {model_name}...")
    try:
        AutoTokenizer.from_pretrained(model_name)
        if model_class:
            model_class.from_pretrained(model_name)
        print(f"  - {model_name} downloaded successfully.")
    except Exception as e:
        print(f"  - Error downloading {model_name}: {e}")

def main():
    print("Starting download of models and datasets to local cache for offline use...")
    
    # 1. Generator Base Model (T5)
    download("t5-base", AutoModelForSeq2SeqLM)
    
    # 2. Reward Model Base (DistilBERT)
    # The 'train_decision_policy' notebook uses this as a base for the Reward Model
    download("distilbert-base-uncased", AutoModelForSequenceClassification)
    
    # 3. BERTScore Base (RoBERTa)
    download("roberta-large")

    # 4. Cache JSON dataset builder
    print("Caching JSON dataset builder script...")
    try:
        with open("dummy.json", "w") as f:
            f.write('{"dummy": "data"}\n')
        load_dataset("json", data_files="dummy.json")
        os.remove("dummy.json")
        print("  - JSON dataset builder cached successfully.")
    except Exception as e:
        print(f"  - Warning: Could not cache json dataset builder: {e}")

    # 5. Evaluation metrics
    print("Caching evaluation metrics...")
    metrics = ["meteor", "rouge", "bleu", "bertscore", "sacrebleu", "chrf", "ter"]
    for m in metrics:
        try:
            evaluate.load(m)
            print(f"  - Cached metric: {m}")
        except:
            pass

    print("\nAll required HuggingFace assets have been requested for download.")
    print("They should now be available in your local cache (~/.cache/huggingface).")
    print("You can now submit your SLURM jobs safely on offline compute nodes.")

if __name__ == "__main__":
    main()
