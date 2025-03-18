import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf
from collections import defaultdict
from minigpt4.models.rec_model import MatrixFactorization

# from baseline import set_random_seed, load_data, create_data_loaders, EarlyStopper, train_one_epoch, evaluate_model, calculate_uauc, train_and_evaluate


def set_random_seed(seed=2023):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir, file_name):
    """Load dataset from a pickle file."""
    file_path = os.path.join(data_dir, file_name)
    return pd.read_pickle(file_path)[['uid', 'iid', 'label']].values


def create_data_loaders(train_data, valid_data, test_data, batch_size):
    """Create PyTorch DataLoaders for training, validation, and testing."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def calculate_uauc(user, predictions, labels):
    """Calculate the user-wise Area Under Curve (uAUC)."""
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    predictions, labels = predictions.squeeze(), labels.squeeze()
    unique_users, inverse_indices, counts = np.unique(user, return_inverse=True, return_counts=True)
    candidates = {user: (predictions[start:start + count], labels[start:start + count])
                  for user, start, count in zip(unique_users, np.cumsum([0] + list(counts[:-1])), counts) if count > 1}

    aucs = []
    for preds, true_labels in candidates.values():
        try:
            aucs.append(roc_auc_score(true_labels, preds))
        except ValueError:
            continue

    uauc = np.mean(aucs) if aucs else 0
    return uauc, len(aucs), len(unique_users) - len(aucs)


class EarlyStopper:
    """Monitor a metric and stop training if no improvement after a given patience."""
    def __init__(self, ref_metric='valid_auc', increase=True, patience=20):
        self.ref_metric = ref_metric
        self.increase = increase
        self.patience = patience
        self.best_metric = None
        self.reach_count = 0

    def update(self, metrics):
        current_value = metrics[self.ref_metric]
        if self.best_metric is None or (self.increase and current_value > self.best_metric[self.ref_metric]) or \
                (not self.increase and current_value < self.best_metric[self.ref_metric]):
            self.best_metric = metrics
            self.reach_count = 0
            return True
        else:
            self.reach_count += 1
            return False

    def should_stop(self):
        return self.reach_count >= self.patience


def evaluate_model(model, data_loader, device):
    """Evaluate the model and calculate metrics."""
    model.eval()
    predictions, labels, users = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            user_ids, item_ids, label = batch[:, 0], batch[:, 1], batch[:, 2]
            scores = model(user_ids.to(device), item_ids.to(device))
            predictions.extend(scores.cpu().numpy())
            labels.extend(label.cpu().numpy())
            users.extend(user_ids.cpu().numpy())
    return predictions, labels, users



def train_one_epoch(model, train_loader, optimizer, criterion, device='mps'):
    """
    Perform one epoch of training.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function for training.
        device (str): Device to run training on. Defaults to 'cpu'.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        # Extract inputs and labels from batch
        user_ids, item_ids, labels = batch[:, 0], batch[:, 1], batch[:, 2]
        
        # Forward pass
        scores = model(user_ids.to(device), item_ids.to(device))
        loss = criterion(scores, labels.float().to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()

    # Return average loss
    return epoch_loss / len(train_loader)


def train_and_evaluate(train_config, data_loaders, model, optimizer, criterion, early_stopper, save_path=None, device='mps'):
    """
    Train and evaluate the model using the given configuration and components.

    Args:
        train_config (dict): Configuration dictionary for training.
        data_loaders (dict): Dictionary containing 'train', 'valid', and 'test' DataLoaders.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.Module): Loss function for training.
        early_stopper (EarlyStopper): Early stopping utility to monitor validation metrics.
        save_path (str, optional): Path to save the best model. Defaults to None.
        device (str): Device to run training on. Defaults to 'cpu'.
    """
    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']

    for epoch in range(train_config['epoch']):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}")

        # Evaluate periodically
        if epoch % train_config['eval_epoch'] == 0:
            predictions, labels, users = evaluate_model(model, valid_loader, device)
            valid_auc = roc_auc_score(labels, predictions)
            valid_uauc, _, _ = calculate_uauc(users, predictions, labels)

            print(f"Epoch {epoch}, Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}")

            # Update early stopper and save the model if improved
            if early_stopper.update({'valid_auc': valid_auc, 'valid_uauc': valid_uauc}):
                if save_path:
                    torch.save(model.state_dict(), save_path)

            if early_stopper.should_stop():
                print("Early stopping triggered.")
                break

    print("Training complete. Best metrics:", early_stopper.best_metric)


def main():
    # Set random seed
    set_random_seed()

    # Load datasets and DataLoaders
    data_dir = "./data/ml-1m/"
    train_data = load_data(data_dir, "train_ood2.pkl")
    valid_data = load_data(data_dir, "valid_ood2.pkl")
    test_data = load_data(data_dir, "test_ood2.pkl")

    # warm or cold
    warm_or_cold = 'warm'
    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])


    data_loaders = {
        'train': DataLoader(train_data, batch_size=1024, shuffle=True),
        'valid': DataLoader(valid_data, batch_size=1024, shuffle=False),
        'test': DataLoader(test_data, batch_size=1024, shuffle=False)
    }

    # Model and optimizer setup
    mf_config = OmegaConf.create({
        "user_num": int(max(train_data[:, 0].max(), valid_data[:, 0].max(), test_data[:, 0].max()) + 1),
        "item_num": int(max(train_data[:, 1].max(), valid_data[:, 1].max(), test_data[:, 1].max()) + 1),
        "embedding_size": 256
    })
    model = MatrixFactorization(mf_config).to('mps')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopper(ref_metric='valid_auc', increase=True, patience=50)

    # Training configuration
    train_config = {
        'lr': 1e-3,
        'wd': 1e-4,
        'embedding_size': 256,
        'epoch': 5000,
        'eval_epoch': 1,
        'patience': 50,
        'batch_size': 1024
    }

    # Train and evaluate
    train_and_evaluate(
        train_config=train_config,
        data_loaders=data_loaders,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        early_stopper=early_stopper,
        save_path="./data/best_mf_model.pth",
        device='mps'
    )


if __name__ == '__main__':
    main()