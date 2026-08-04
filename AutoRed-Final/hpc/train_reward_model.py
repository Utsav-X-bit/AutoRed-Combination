import os
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import requests

def train_reward_model(
    data_path="https://github.com/HumanCompatibleAI/tensor-trust-data/raw/main/detecting-extractions/v1/prompt_extraction_detection.jsonl",
    model_name='distilbert-base-uncased',
    output_dir='pre_trained/pi_reward_model',
    epochs=3,
    batch_size=64,
    lr=5e-5,
    device=os.environ.get('CUDA_VISIBLE_DEVICES', 'cuda' if torch.cuda.is_available() else 'cpu')
):
    if str(device).isdigit() or (isinstance(device, str) and ',' in device):
        device = 'cuda'
        
    print(f"Starting Reward Model (Stop Point Identifier) training on {device}...")
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset (if not available locally, download)
    local_data = "prompt_extraction_detection.jsonl"
    if not os.path.exists(local_data):
        print(f"Downloading dataset from {data_path}...")
        r = requests.get(data_path)
        with open(local_data, 'wb') as f:
            f.write(r.content)
    
    df = pd.read_json(local_data, lines=True)
    
    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Preprocessing
    def preprocess(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Simple split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    def get_loader(dataset_df):
        encodings = preprocess(dataset_df['llm_output'].tolist())
        labels = torch.tensor(dataset_df['is_prompt_extraction'].astype(int).tolist())
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = get_loader(train_df)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids, mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Average loss: {total_loss / len(train_loader)}")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Reward Model saved to {output_dir}")

if __name__ == "__main__":
    train_reward_model()
