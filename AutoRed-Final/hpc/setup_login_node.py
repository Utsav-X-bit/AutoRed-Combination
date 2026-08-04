import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

def download(model_name, model_class=None):
    print(f"Downloading {model_name}...")
    try:
        AutoTokenizer.from_pretrained(model_name)
        if model_class:
            model_class.from_pretrained(model_name)
        print(f"  - {model_name} ready.")
    except Exception as e:
        print(f"  - Error with {model_name}: {e}")

def main():
    # Base models
    download("t5-base", AutoModelForSeq2SeqLM)
    download("distilbert-base-uncased", AutoModelForSequenceClassification)
    download("roberta-large")

    # Metrics
    metrics = ["meteor", "rouge", "bleu", "bertscore", "sacrebleu", "chrf", "ter"]
    for m in metrics:
        try:
            evaluate.load(m)
            print(f"  - Metric {m} cached.")
        except: pass

    # Dataset builder
    try:
        load_dataset("json", data_files={"train": "scripts/pi/pi_data/pi_gen_data/train.json"}, split="train")
    except: pass

if __name__ == "__main__":
    main()
