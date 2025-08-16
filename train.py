from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import wandb
from config import CFG as cfg

# Output directories based on configured SKELETON_DIR
SAVED_MODELS_DIR = os.path.join(cfg.SKELETON_DIR, "saved_models")
WANDB_DIR = os.path.join(cfg.SKELETON_DIR, "wandb")

# Ensure output directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(WANDB_DIR, exist_ok=True)

# Select computation device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)
print(f"Using device: {device}")


def plot_confusion_matrix(cm, class_names):
    """
    Plot a row-normalized confusion matrix for classification results.

    Args:
        cm (ndarray): Confusion matrix (n_classes x n_classes)
        class_names (list[str]): Class labels in index order
    """
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]  # Normalize per row
    df_cm = pd.DataFrame(cm, class_names, class_names)    # Create DataFrame for plotting
    ax = sn.heatmap(df_cm, annot=True, cmap="flare", fmt=".2f")  # Heatmap visualization
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()


def count_classes(preds):
    """
    Count predicted class occurrences in a batch.

    Args:
        preds (Tensor): Logits of shape [batch_size, n_classes]
    Returns:
        list[int]: Count per class index
    """
    pred_classes = preds.argmax(dim=1)  # Convert logits to predicted class indices
    return [(pred_classes == c).sum().item() for c in range(preds.shape[1])]


def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    """
    Train the model for one epoch.

    Returns:
        dict: {'Loss_train', 'Accuracy_train', 'F1_train'}
    """
    model.train()  # Enable training mode (dropout, BN updates)

    # Define metrics for tracking
    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    # Progress bar for training
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for inputs, labels in pbar:
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_metric.update(loss)
        acc_metric.update(outputs, labels.long())
        f1_metric.update(outputs, labels.long())

        # Update live progress bar
        pbar.set_postfix({
            "Loss": f"{loss_metric.compute():.4f}",
            "Acc":  f"{acc_metric.compute():.4f}",
            "F1":   f"{f1_metric.compute():.4f}",
        })

    return {
        "Loss_train": loss_metric.compute(),
        "Accuracy_train": acc_metric.compute(),
        "F1_train": f1_metric.compute(),
    }


def val_epoch(epoch, model, criterion, loader, num_classes, device):
    """
    Validate the model for one epoch.

    Returns:
        (metrics_dict, confusion_matrix)
        where metrics_dict: {'Loss_val','Accuracy_val','F1_val'}
    """
    model.eval()  # Evaluation mode (no dropout, BN frozen)

    # Define metrics for tracking
    loss_metric = torchmetrics.MeanMetric().to(device)
    acc_metric  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    cm_metric   = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    with torch.no_grad():  # Disable gradients for validation
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            # Update metrics
            loss_metric.update(loss)
            acc_metric.update(outputs, labels.long())
            f1_metric.update(outputs, labels.long())
            cm_metric.update(preds, labels.long())

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss_metric.compute():.4f}",
                "Acc":  f"{acc_metric.compute():.4f}",
                "F1":   f"{f1_metric.compute():.4f}",
            })

    # Collect results
    metrics = {
        "Loss_val": loss_metric.compute(),
        "Accuracy_val": acc_metric.compute(),
        "F1_val": f1_metric.compute(),
    }
    cm = cm_metric.compute().detach().cpu().numpy()  # Convert to NumPy for plotting
    return metrics, cm


def train_model(model, train_loader, val_loader, optimizer, criterion, class_names,
                n_epochs, project_name, ident_str=None):
    """
    Full training loop with per-epoch validation, W&B logging, and checkpointing.

    Returns:
        str: Path to saved weights file
    """
    num_classes = len(class_names)
    model.to(device)  # Move model to target device

    # Generate unique run name
    if ident_str is None:
        ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"

    # Initialize W&B logging in custom directory
    run = wandb.init(
        project=project_name,
        name=exp_name,
        dir=WANDB_DIR
    )

    try:
        for epoch in range(n_epochs):
            # Training phase
            train_metrics = train_epoch(epoch, model, optimizer, criterion,
                                        train_loader, num_classes, device)

            # Validation phase
            val_metrics, cm = val_epoch(epoch, model, criterion,
                                        val_loader, num_classes, device)

            # Log metrics to W&B
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch + 1})
    finally:
        run.finish()

    # Plot and display confusion matrix after training
    plot_confusion_matrix(cm, class_names)

    # Save trained model weights
    model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    return model_path
