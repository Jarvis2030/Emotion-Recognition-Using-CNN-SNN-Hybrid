"""
Training script for the CNN-SNN model.

Adapt the `load_data()` function to your dataset. The rest of the pipeline
(training loop, validation, early saving, testing) is general-purpose.

Requirements:
    torch, snntorch, numpy, scikit-learn
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import snntorch.functional
import numpy as np
from pathlib import Path

from ref_cnn_snn_model import CnnSnn


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "fs": 256,                # Sampling frequency (Hz)
    "decision_window": 1,     # Window duration (seconds)
    "out_channels": 50,       # Number of conv filters / hidden units
    "dropout": 0.5,           # Dropout probability
    "batch_size": 64,
    "lr": 1e-3,
    "num_epochs": 1000,
    "save_dir": "./checkpoints",
}


# =============================================================================
# Data loading — MODIFY THIS FOR YOUR DATASET
# =============================================================================

def load_data():
    """Load and return train/valid/test splits as TensorDatasets.

    Expected tensor shapes:
        data:   (num_samples, num_channels, num_timepoints)
        labels: (num_samples,) with integer class labels

    Returns:
        train_dataset: TensorDataset
        valid_dataset: TensorDataset
        test_dataset:  TensorDataset
        in_channels:   int, number of input channels
    """
    raise NotImplementedError(
        "Implement load_data() for your dataset. "
        "Return (train_dataset, valid_dataset, test_dataset, in_channels). "
        "Each dataset should be a TensorDataset of (float32 data, long labels)."
    )

    # ---- Example using a .mat file ----
    # import mat73
    # mat = mat73.loadmat("your_data.mat")["data_key"]
    #
    # train_data = torch.from_numpy(mat["train"]).permute(0, 2, 1).float()
    # train_labels = torch.from_numpy(mat["train_labels"]).long()
    # ... same for valid and test ...
    #
    # in_channels = train_data.shape[1]
    # return (
    #     TensorDataset(train_data, train_labels),
    #     TensorDataset(valid_data, valid_labels),
    #     TensorDataset(test_data, test_labels),
    #     in_channels,
    # )


# =============================================================================
# Training & evaluation
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch. Returns total loss."""
    model.train()
    total_loss = 0.0

    for data, labels in dataloader:
        data = data.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        _, spike_rec, _ = model(data)
        loss = criterion(spike_rec, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in dataloader:
        data = data.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        _, spike_rec, _ = model(data)
        total_loss += criterion(spike_rec, labels).item()

        total += labels.size(0)
        correct += snntorch.functional.accuracy_rate(spike_rec, labels) * labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return total_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_ds, valid_ds, test_ds, in_channels = load_data()

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # Initialize model
    model = CnnSnn(
        fs=CONFIG["fs"],
        decision_window=CONFIG["decision_window"],
        in_channels=in_channels,
        out_channels=CONFIG["out_channels"],
        dropout=CONFIG["dropout"],
    ).to(device)
    print(model)

    criterion = snntorch.functional.loss.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Training loop with best-model checkpointing
    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"

    best_valid_acc = 0.0

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        _, train_acc = evaluate(model, train_loader, criterion, device)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_train_acc = train_acc
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f}  Acc: {valid_acc:.4f}"
        )

    print(f"\nBest validation accuracy: {best_valid_acc:.4f}")
    print(f"Corresponding train accuracy: {best_train_acc:.4f}")

    # Test with best model
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()