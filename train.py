import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 配置
# 模型: Microsoft CodeBERT (Base version)
MODEL_NAME = "microsoft/codebert-base"
DATA_FILE = 'vscode_dataset_clean.csv'
SAVE_PATH = './saved_model'

# 参数
MAX_LEN = 128          
BATCH_SIZE = 16        
EPOCHS = 3             
LEARNING_RATE = 2e-5  
SEED = 42             

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

class IssueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model = model.train()
    losses = []
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=loss.item())

    return np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())

    return classification_report(real_values, predictions, target_names=['Bug', 'Feature']), accuracy_score(real_values, predictions)

def main():
    set_seed(SEED)
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file {DATA_FILE} not found.")
        
    df = pd.read_csv(DATA_FILE)
    print(f"Data Loaded. Shape: {df.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['body'].values, 
        df['label'].values, 
        test_size=0.15, 
        random_state=SEED
    )
    
    # 2. Tokenizer
    print("Loading CodeBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = IssueDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = IssueDataset(X_test, y_test, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 3. Model Initialization
    print("Loading CodeBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    # Optimizer and Scheduler
    # 删掉 correct_bias 参数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 4. Training Loop
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        report, accuracy = eval_model(model, test_loader, device)
        print(f"Val Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存最优模型
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            
            print(f"New best model! Saving to {SAVE_PATH}...")
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            
    print("\nTraining Complete.")
    print(f"Best Accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()
# TODO: 这里还没写完，明天继续
