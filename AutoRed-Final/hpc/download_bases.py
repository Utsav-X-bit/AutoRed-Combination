import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

def main():
    print("Starting download of base models for Defense Classifier and Ranker...")
    
    models = [
        ("distilbert-base-uncased", AutoModelForSequenceClassification),
        ("microsoft/deberta-v3-base", AutoModel)
    ]
    
    for model_name, model_class in models:
        print(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name)
            model_class.from_pretrained(model_name)
            print(f"✓ {model_name} downloaded successfully.")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")
            
    print("Download script finished.")

if __name__ == "__main__":
    main()
