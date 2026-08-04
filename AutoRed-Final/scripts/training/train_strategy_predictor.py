#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
EXP_DIR = ROOT_DIR / "experiment"

TRAIN_DATA = DATA_DIR / "strategy_matrix_raw_v1.jsonl"
FEATURE_VOCAB_PATH = EXP_DIR / "feature_vocab.json"
LABEL_VOCAB_PATH = EXP_DIR / "label_vocab.json"
MODEL_PATH = EXP_DIR / "strategy_predictor.pth"

# All 18 strategies
ATTACK_TYPES = [
    "instruction_leak",
    "trigger_phrase_discovery",
    "exception_discovery",
    "roleplay",
    "translation",
    "summarization",
    "system_prompt_recovery",
    "encoding_bypass",
    "markdown_smuggling",
    "latent_injection",
    "authority_override",
    "jailbreak_framing",
    "reflection_attack",
    "format_conversion",
    "json_smuggling",
    "yaml_smuggling",
    "base64_bypass",
    "unicode_bypass"
]

class StrategyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.stack(X)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StrategyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def build_vocabs(data_path):
    features = set()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            features.add(f"primary:{item.get('primary_type', 'UNKNOWN')}")
            for sec in item.get('secondary_flags', []):
                features.add(f"secondary:{sec}")
            # Add new features that were not present previously
            features.add(f"code_type:{item.get('access_code_type', 'UNKNOWN')}")
            features.add(f"complexity:{item.get('defense_complexity', 'medium')}")
                
    feature_vocab = {f: i for i, f in enumerate(sorted(list(features)))}
    label_vocab = {l: i for i, l in enumerate(ATTACK_TYPES)}
    
    return feature_vocab, label_vocab

def load_data(data_path, feature_vocab, label_vocab):
    X_all = []
    y_all = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Target is the winning strategy
            if not item.get("success"):
                continue
                
            strategy = item.get("strategy_used")
            if strategy not in label_vocab:
                continue
                
            feat_vec = torch.zeros(len(feature_vocab))
            
            prim = f"primary:{item.get('primary_type', 'UNKNOWN')}"
            if prim in feature_vocab:
                feat_vec[feature_vocab[prim]] = 1.0
                
            for sec in item.get('secondary_flags', []):
                sec_feat = f"secondary:{sec}"
                if sec_feat in feature_vocab:
                    feat_vec[feature_vocab[sec_feat]] = 1.0
                    
            ctype = f"code_type:{item.get('access_code_type', 'UNKNOWN')}"
            if ctype in feature_vocab:
                feat_vec[feature_vocab[ctype]] = 1.0
                
            comp = f"complexity:{item.get('defense_complexity', 'medium')}"
            if comp in feature_vocab:
                feat_vec[feature_vocab[comp]] = 1.0
                
            X_all.append(feat_vec)
            y_all.append(label_vocab[strategy])
            
    return X_all, y_all

def main():
    print("Building vocabularies...")
    feature_vocab, label_vocab = build_vocabs(TRAIN_DATA)
    
    with open(FEATURE_VOCAB_PATH, "w") as f:
        json.dump(feature_vocab, f, indent=2)
    with open(LABEL_VOCAB_PATH, "w") as f:
        json.dump(label_vocab, f, indent=2)
        
    print(f"Feature vocab size: {len(feature_vocab)}")
    print(f"Label vocab size: {len(label_vocab)}")
    
    print("Loading dataset...")
    X_all, y_all = load_data(TRAIN_DATA, feature_vocab, label_vocab)
    print(f"Total successful attempts loaded: {len(X_all)}")
    
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    train_dataset = StrategyDataset(X_train, y_train)
    val_dataset = StrategyDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Compute class weights
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = len(label_vocab)
    weights = torch.ones(num_classes)
    for cls_idx in range(num_classes):
        if class_counts[cls_idx] > 0:
            weights[cls_idx] = total_samples / (num_classes * class_counts[cls_idx])
            
    model = StrategyPredictor(len(feature_vocab), len(label_vocab))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    epochs = 20
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            
        train_acc = correct / len(train_dataset)
        
        # Eval
        model.eval()
        val_correct = 0
        val_correct_top3 = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                
                # Top-3
                _, top3_preds = logits.topk(3, dim=1)
                val_correct_top3 += sum([1 for i in range(len(y_batch)) if y_batch[i] in top3_preds[i]])
                
        val_acc = val_correct / len(val_dataset)
        val_top3_acc = val_correct_top3 / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Top-3: {val_top3_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best model! Saving to {MODEL_PATH}")
            torch.save(model.state_dict(), MODEL_PATH)
            
    print("Done!")

if __name__ == "__main__":
    main()
