#!/usr/bin/env python3
"""Train a DeBERTa-v3-base binary classifier for extraction candidate ranking.

Replaces AutoRed's hardcoded 0.35·LLM + 0.25·Regex + ... formula with a learned
discriminator that takes (victim_response, candidate, access_code_type) and outputs
P(candidate is the correct secret).

Usage:
    python scripts/training/train_ranker.py \
        --train-file data/ranker_dataset_train_v1.jsonl \
        --val-file data/ranker_dataset_val_v1.jsonl \
        --test-file data/ranker_dataset_test_v1.jsonl \
        --output-dir models/ranker_deberta_v1/
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch import nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_inputs(records: list[dict]) -> tuple[list[str], list[int], list[str]]:
    """Prepare model inputs from records."""
    texts = []
    labels = []
    types = []
    for r in records:
        vr = r.get("victim_response", "")[:1500]  # Truncate long responses
        cand = r.get("candidate", "")[:200]
        ac_type = r.get("access_code_type", "UNKNOWN")
        # Format: victim_response [SEP] candidate [SEP] Type: access_code_type
        text = f"{vr} [SEP] {cand} [SEP] Type: {ac_type}"
        texts.append(text)
        labels.append(r.get("label", 0))
        types.append(ac_type)
    return texts, labels, types


def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
    }


class WeightedTrainer:
    """Custom Trainer that uses class-weighted CrossEntropyLoss."""

    @staticmethod
    def create(base_trainer_cls, class_weights):
        """Create a Trainer subclass with weighted loss."""
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        class _WeightedTrainer(base_trainer_cls):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fn = nn.CrossEntropyLoss(
                    weight=weights_tensor.to(logits.device)
                )
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        return _WeightedTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa extractor ranker")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--train-file", default="data/ranker_dataset_train_v1.jsonl")
    parser.add_argument("--val-file", default="data/ranker_dataset_val_v1.jsonl")
    parser.add_argument("--test-file", default="data/ranker_dataset_test_v1.jsonl")
    parser.add_argument("--output-dir", default="models/ranker_deberta_v1/")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    # Lazy imports (heavy)
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    log.info(f"Loading training data from {args.train_file}...")
    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    test_records = load_jsonl(args.test_file)

    log.info(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # Prepare inputs
    train_texts, train_labels, train_types = prepare_inputs(train_records)
    val_texts, val_labels, val_types = prepare_inputs(val_records)
    test_texts, test_labels, test_types = prepare_inputs(test_records)

    # Compute class weights
    label_counts = Counter(train_labels)
    total = sum(label_counts.values())
    n_classes = 2
    class_weights = [
        total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)
    ]
    log.info(f"Label distribution: {dict(label_counts)}")
    log.info(f"Class weights: {class_weights}")

    # Load tokenizer and model
    log.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Tokenize
    def tokenize_texts(texts, labels):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors=None,
        )
        encodings["labels"] = labels
        return Dataset.from_dict(encodings)

    log.info("Tokenizing datasets...")
    train_dataset = tokenize_texts(train_texts, train_labels)
    val_dataset = tokenize_texts(val_texts, val_labels)
    test_dataset = tokenize_texts(test_texts, test_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
    )

    # Create weighted trainer
    WeightedTrainerCls = WeightedTrainer.create(Trainer, class_weights)

    trainer = WeightedTrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    log.info("Starting training...")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save best model + tokenizer
    log.info(f"Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate on test set
    log.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_probs = torch.softmax(
        torch.tensor(predictions.predictions), dim=-1
    )[:, 1].numpy()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(test_labels, pred_labels)
    report = classification_report(
        test_labels, pred_labels, target_names=["Negative", "Positive"]
    )

    # Per-type analysis
    per_type_metrics = {}
    for ac_type in set(test_types):
        type_mask = [t == ac_type for t in test_types]
        type_labels = [l for l, m in zip(test_labels, type_mask) if m]
        type_preds = [p for p, m in zip(pred_labels.tolist(), type_mask) if m]
        if type_labels:
            from sklearn.metrics import (
                precision_recall_fscore_support as prfs,
                accuracy_score as acc,
            )

            p, r, f, _ = prfs(
                type_labels, type_preds, average="binary", zero_division=0
            )
            per_type_metrics[ac_type] = {
                "count": len(type_labels),
                "accuracy": acc(type_labels, type_preds),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }

    # Save results
    results = {
        "test_metrics": {k: float(v) for k, v in test_results.items()},
        "confusion_matrix": cm.tolist(),
        "per_type_metrics": per_type_metrics,
        "model_name": args.model_name,
        "total_params": total_params,
        "class_weights": class_weights,
    }
    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RANKER TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Model: {args.model_name}")
    print(f"  Params: {total_params:,}")
    print(f"\n  Test Results:")
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"    {k:30s}: {v:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(f"\n  Classification Report:\n{report}")
    print(f"\n  Per-Type Metrics:")
    for t, m in sorted(per_type_metrics.items()):
        print(
            f"    {t:20s}: F1={m['f1']:.3f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} (n={m['count']})"
        )
    print(f"\n  Model saved to: {args.output_dir}")
    print(f"  Results saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
