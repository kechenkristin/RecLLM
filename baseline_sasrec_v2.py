import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf
from minigpt4.models.rec_base_models import SASRec

# ==============================
# Utility Functions
# ==============================

def set_random_seed(seed: int = 2023):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir: str, filename: str):
    """Loads dataset from a pickle file."""
    return pd.read_pickle(os.path.join(data_dir, filename))[['uid', 'iid', 'his', 'label']].values

def calculate_uauc(user, predictions, labels):
    """Calculates user-wise AUC."""
    predictions, labels = np.array(predictions).squeeze(), np.array(labels).squeeze()
    unique_users, inverse_indices, counts = np.unique(user, return_inverse=True, return_counts=True)

    aucs = []
    for user_id in unique_users:
        indices = np.where(user == user_id)
        user_preds, user_labels = predictions[indices], labels[indices]
        try:
            aucs.append(roc_auc_score(user_labels, user_preds))
        except ValueError:
            continue

    return np.mean(aucs) if aucs else 0

# ==============================
# Dataset Classes
# ==============================

class SequenceDataset(Dataset):
    """Dataset class for sequential data."""
    def __init__(self, data, max_len=25):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid, iid, history, label = self.data[index]
        history = np.pad(history[-self.max_len:], (self.max_len - len(history), 0), 'constant') if len(history) < self.max_len else np.array(history[-self.max_len:])
        return uid, iid, history, label

# ==============================
# Training & Evaluation Logic
# ==============================

class EarlyStopper:
    """Class for early stopping."""
    def __init__(self, patience=20, metric="valid_auc", higher_is_better=True):
        self.metric = metric
        self.best_score = None
        self.patience = patience
        self.counter = 0
        self.higher_is_better = higher_is_better

    def update(self, score):
        """Updates early stopping counter and checks for stopping condition."""
        if self.best_score is None or (self.higher_is_better and score > self.best_score) or (not self.higher_is_better and score < self.best_score):
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        """Returns True if training should stop."""
        return self.counter >= self.patience

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Performs one training epoch."""
    model.train()
    epoch_loss = 0.0

    for batch in data_loader:
        user_ids, item_ids, history, labels = (x.to(device) for x in batch)
        outputs = model(history.long(), item_ids.long())
        loss = criterion(outputs, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    """Evaluates the model."""
    model.eval()
    predictions, labels, users = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            user_ids, item_ids, history, batch_labels = (x.to(device) for x in batch)
            outputs = model.forward_eval(user_ids.long(), item_ids.long(), history.long())
            predictions.extend(outputs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            users.extend(user_ids.cpu().numpy())

    auc = roc_auc_score(labels, predictions)
    uauc = calculate_uauc(np.array(users), np.array(predictions), np.array(labels))

    return auc, uauc

def train_and_evaluate(config, data_loaders, model, optimizer, criterion, early_stopper, device="mps"):
    """Train and evaluate the model."""
    train_loader, valid_loader, test_loader = data_loaders["train"], data_loaders["valid"], data_loaders["test"]

    for epoch in range(config["epoch"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        if epoch % config["eval_epoch"] == 0:
            valid_auc, valid_uauc = evaluate_model(model, valid_loader, device)
            print(f"Epoch {epoch}, Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}")

            if early_stopper.update(valid_auc):
                torch.save(model.state_dict(), config["save_path"])
            
            if early_stopper.should_stop():
                print("Early stopping triggered.")
                break

    print("Training complete. Best AUC:", early_stopper.best_score)

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    set_random_seed()

    # Configuration
    config = {
        "lr": 1e-2,
        "wd": 1e-4,
        "embedding_size": 64,
        "epoch": 5000,
        "eval_epoch": 1,
        "patience": 50,
        "batch_size": 1024,
        "maxlen": 25,
        "save_path": "./data/best_sasrec_model.pth",
    }

    # Load datasets
    data_dir = "./data/ml-1m/"
    train_data = load_data(data_dir, "train_ood2.pkl")
    valid_data = load_data(data_dir, "valid_ood2.pkl")
    test_data = load_data(data_dir, "test_ood2.pkl")

    # Create dataset instances
    datasets = {
        "train": SequenceDataset(train_data, max_len=config["maxlen"]),
        "valid": SequenceDataset(valid_data, max_len=config["maxlen"]),
        "test": SequenceDataset(test_data, max_len=config["maxlen"]),
    }

    # Create DataLoaders
    data_loaders = {key: DataLoader(dataset, batch_size=config["batch_size"], shuffle=(key == "train")) for key, dataset in datasets.items()}

    # Model configuration
    user_num = max(train_data[:, 0].max(), valid_data[:, 0].max(), test_data[:, 0].max()) + 1
    item_num = max(train_data[:, 1].max(), valid_data[:, 1].max(), test_data[:, 1].max()) + 1

    sasrec_config = OmegaConf.create({
        "user_num": user_num,
        "item_num": item_num,
        "hidden_units": config["embedding_size"],
        "num_blocks": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "l2_emb": 1e-4,
        "maxlen": config["maxlen"],
    })

    # Initialize Model
    model = SASRec(sasrec_config).to("mps")

    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopper
    early_stopper = EarlyStopper(patience=config["patience"], metric="valid_auc")

    # Train and Evaluate
    train_and_evaluate(config, data_loaders, model, optimizer, criterion, early_stopper, device="mps")