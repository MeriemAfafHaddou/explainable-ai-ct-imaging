import torch


def enable_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model,
    images,
    device,
    n_samples=20,
):
    """Estimate predictions and uncertainty using MC Dropout."""
    model.eval()
    enable_dropout(model)

    probabilities = []

    images = images.to(device)

    for _ in range(n_samples):
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(probs)

    probabilities = torch.stack(probabilities)

    mean_probability = probabilities.mean(dim=0)
    uncertainty = probabilities.std(dim=0)

    predictions = (mean_probability >= 0.5).long()

    return {
        "probabilities": mean_probability,
        "uncertainty": uncertainty,
        "predictions": predictions,
    }


def collect_uncertainty_predictions(
    model,
    data_loader,
    device,
    n_samples=20,
):
    model.eval()

    all_labels = []
    all_probabilities = []
    all_uncertainties = []
    all_predictions = []

    for images, labels in data_loader:
        results = mc_dropout_predict(
            model=model,
            images=images,
            device=device,
            n_samples=n_samples,
        )

        all_labels.extend(labels.numpy())
        all_probabilities.extend(
            results["probabilities"].cpu().numpy()
        )
        all_uncertainties.extend(
            results["uncertainty"].cpu().numpy()
        )
        all_predictions.extend(
            results["predictions"].cpu().numpy()
        )

    return {
        "labels": all_labels,
        "probabilities": all_probabilities,
        "uncertainties": all_uncertainties,
        "predictions": all_predictions,
    }
