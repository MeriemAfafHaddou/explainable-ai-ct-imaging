from copy import deepcopy

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _compute_classification_metrics(
    all_labels,
    all_predictions,
    all_probabilities,
):
    """Compute the core classification metrics."""
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    _, _, fn, tp = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if len(set(all_labels)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels, all_probabilities)

    return {
        "f1": f1,
        "sensitivity": sensitivity,
        "auroc": auroc,
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        predictions = outputs.argmax(dim=1)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

    train_loss = total_loss / total_samples

    metrics = _compute_classification_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )
    metrics["loss"] = train_loss

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    """Evaluate the model and compute validation metrics."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_logits = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        predictions = outputs.argmax(dim=1)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_logits.extend(outputs.cpu().numpy())

    validation_loss = total_loss / total_samples

    metrics = _compute_classification_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    metrics["loss"] = validation_loss
    metrics["labels"] = all_labels
    metrics["probabilities"] = all_probabilities
    metrics["logits"] = all_logits

    return metrics


def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    patience=5,
    checkpoint_metric="f1",
):
    """Train a model with learning-rate scheduling and early stopping.

    The best checkpoint is selected using the validation metric specified
    by checkpoint_metric. Validation labels, probabilities, and logits from
    the best epoch are stored for later calibration analysis.
    """
    best_val_metric = float("-inf")
    best_model_state = None
    best_validation_predictions = None
    best_epoch = None
    epochs_without_improvement = 0

    epoch_states = {}

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_f1": [],
        "train_sensitivity": [],
        "train_auroc": [],
        "f1": [],
        "sensitivity": [],
        "auroc": [],
    }

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        train_loss = train_metrics["loss"]
        validation_loss = validation_metrics["loss"]

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_sensitivity"].append(train_metrics["sensitivity"])
        history["train_specificity"].append(train_metrics["specificity"])
        history["train_auroc"].append(train_metrics["auroc"])

        history["accuracy"].append(validation_metrics["accuracy"])
        history["f1"].append(validation_metrics["f1"])
        history["sensitivity"].append(validation_metrics["sensitivity"])
        history["specificity"].append(validation_metrics["specificity"])
        history["auroc"].append(validation_metrics["auroc"])

        epoch_states[epoch + 1] = deepcopy(model.state_dict())

        current_metric = validation_metrics[checkpoint_metric]

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_model_state = deepcopy(model.state_dict())

            best_validation_predictions = {
                "labels": validation_metrics["labels"],
                "probabilities": validation_metrics["probabilities"],
                "logits": validation_metrics["logits"],
            }

            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(validation_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val F1: {validation_metrics['f1']:.4f} | "
            f"Val Sensitivity: {validation_metrics['sensitivity']:.4f} | "
            f"Val AUROC: {validation_metrics['auroc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping. Best epoch: {best_epoch} "
                f"({checkpoint_metric}={best_val_metric:.4f})"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        raise RuntimeError(
            "No best checkpoint was captured during training. "
            "Check that validation metrics are being computed correctly."
        )

    return (
        model,
        history,
        best_validation_predictions,
        epoch_states,
        best_epoch,
    )
