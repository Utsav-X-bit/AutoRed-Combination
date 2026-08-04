import os
import json
import torch
import math
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/access_code_classifier_dataset.jsonl"
OUTPUT_DIR = "experiment/access_code_predictor"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

LABEL_MAP = {
    "TOKEN": 0,
    "MULTILINE": 1,
    "PHRASE": 2,
    "SENTENCE": 3
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class AccessCodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Custom Trainer to apply class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2, 3], zero_division=0)
    
    macro_f1 = f1.mean()
    
    metrics = {"macro_f1": macro_f1}
    for i, name in INV_LABEL_MAP.items():
        metrics[f"{name}_precision"] = precision[i]
        metrics[f"{name}_recall"] = recall[i]
        metrics[f"{name}_f1"] = f1[i]
        
    return metrics

def train():
    print("Loading dataset...")
    texts = []
    labels = []
    counts = {0:0, 1:0, 2:0, 3:0}
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = f"{data['opening_defense']} [SEP] {data['closing_defense']}"
            label = LABEL_MAP[data["access_code_type"]]
            texts.append(text)
            labels.append(label)
            counts[label] += 1
            
    # Calculate smoothed class weights: weight = sqrt(max_count / class_count)
    max_count = max(counts.values())
    weights = [math.sqrt(max_count / counts[i]) for i in range(len(counts))]
    print("\nClass Counts:", counts)
    print("Smoothed Weights:")
    for i, w in enumerate(weights):
        print(f"  {INV_LABEL_MAP[i]}: {w:.2f}")
        
    class_weights = torch.tensor(weights, dtype=torch.float)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print("\nInitializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    )
    
    train_dataset = AccessCodeDataset(train_texts, train_labels, tokenizer)
    val_dataset = AccessCodeDataset(val_texts, val_labels, tokenizer)
    
    try:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=128,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none"
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=128,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none"
        )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining complete. Evaluating on validation set...")
    val_results = trainer.predict(val_dataset)
    
    # Save Confusion Matrix
    print("\nComputing confusion matrix...")
    preds = val_results.predictions.argmax(-1)
    cm = confusion_matrix(val_labels, preds, labels=[0, 1, 2, 3])
    cm_dict = {
        "labels": ["TOKEN", "MULTILINE", "PHRASE", "SENTENCE"],
        "matrix": cm.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm_dict, f, indent=2)
        
    print(cm)
    
    # Temperature Scaling for Calibration
    print("\nCalibrating probabilities (Temperature Scaling)...")
    logits = torch.tensor(val_results.predictions)
    val_labels_t = torch.tensor(val_labels)
    
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()
    
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, val_labels_t)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    opt_temp = temperature.item()
    print(f"Optimal Temperature: {opt_temp:.4f}")
    
    # Save the calibrated model temperature
    model.config.temperature = opt_temp
    model.config.id2label = INV_LABEL_MAP
    model.config.label2id = LABEL_MAP
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nModel and calibration temp saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
