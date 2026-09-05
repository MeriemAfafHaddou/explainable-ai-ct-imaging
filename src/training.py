from copy import deepcopy

import torch


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

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    """Evaluate the model on the validation set."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


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
):
    """Train a model with learning-rate scheduling and early stopping."""
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        scheduler.step(validation_loss)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

            # Keep the model from the epoch with the best validation performance.
            best_model_state = deepcopy(model.state_dict())

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f}"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
