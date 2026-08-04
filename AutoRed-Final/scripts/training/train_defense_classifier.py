import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }

def main():
    data_path = "data/defense_classifier_dataset.jsonl"
    model_name = "distilbert-base-uncased"
    output_dir = "models/defense_classifier"
    
    print(f"Loading data from {data_path}...")
    texts = []
    labels_str = []
    
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Combine opening and closing defense
            text = f"{data['opening_defense']} [SEP] {data['closing_defense']}"
            texts.append(text)
            labels_str.append(data['defense_type'])
            
    # Create label mapping
    unique_labels = sorted(list(set(labels_str)))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"Found {len(unique_labels)} classes: {unique_labels}")
    
    labels = [label2id[l] for l in labels_str]
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)
        
    print("Tokenizing data...")
    train_encodings = tokenize(train_texts)
    val_encodings = tokenize(val_texts)
    
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving best model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save the label mapping
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
        
    print("Training complete! Model saved to", output_dir)

if __name__ == "__main__":
    main()
