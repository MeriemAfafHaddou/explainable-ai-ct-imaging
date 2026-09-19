import numpy as np
import torch
from scipy.stats import beta, norm
from sklearn.metrics import confusion_matrix


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
        all_probabilities.extend(results["probabilities"].cpu().numpy())
        all_uncertainties.extend(results["uncertainty"].cpu().numpy())
        all_predictions.extend(results["predictions"].cpu().numpy())

    return {
        "labels": all_labels,
        "probabilities": all_probabilities,
        "uncertainties": all_uncertainties,
        "predictions": all_predictions,
    }


def expected_calibration_error(labels, probabilities, n_bins=10):
    predictions = (probabilities >= 0.5).astype(int)

    confidence = np.where(
        predictions == 1,
        probabilities,
        1 - probabilities,
    )

    correctness = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)

        if np.any(in_bin):
            accuracy = correctness[in_bin].mean()
            mean_confidence = confidence[in_bin].mean()
            bin_size = in_bin.sum()

            ece += (bin_size / len(labels)) * abs(accuracy - mean_confidence)

    return ece


def wilson_ci(successes, total, confidence=0.95):
    """Return a Wilson confidence interval for a proportion."""
    if total == 0:
        return np.nan, np.nan

    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)

    p = successes / total

    denominator = 1 + (z**2 / total)

    center = (p + (z**2 / (2 * total))) / denominator

    margin = z * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2))) / denominator

    return center - margin, center + margin


def clopper_pearson_ci(successes, total, confidence=0.95):
    """Return an exact Clopper-Pearson confidence interval."""
    if total == 0:
        return np.nan, np.nan

    alpha = 1 - confidence

    if successes == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, successes, total - successes + 1)

    if successes == total:
        upper = 1.0
    else:
        upper = beta.ppf(
            1 - alpha / 2,
            successes + 1,
            total - successes,
        )

    return lower, upper


def patient_level_bootstrap_ci(
    labels,
    predictions,
    patient_ids,
    metric,
    n_bootstrap=2000,
    confidence=0.95,
    random_state=42,
):
    """Estimate a patient-level bootstrap CI for a metric."""
    rng = np.random.default_rng(random_state)

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    patient_ids = np.asarray(patient_ids)

    unique_patients = np.unique(patient_ids)

    bootstrap_values = []

    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [np.where(patient_ids == patient_id)[0] for patient_id in sampled_patients]
        )

        sampled_labels = labels[sampled_indices]
        sampled_predictions = predictions[sampled_indices]

        value = metric(
            sampled_labels,
            sampled_predictions,
        )

        if not np.isnan(value):
            bootstrap_values.append(value)

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_values,
        100 * (alpha / 2),
    )

    upper = np.percentile(
        bootstrap_values,
        100 * (1 - alpha / 2),
    )

    return lower, upper


def sensitivity_score(labels, predictions):
    """Compute sensitivity from binary labels and predictions."""
    _, _, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    denominator = tp + fn

    return tp / denominator if denominator > 0 else np.nan


def specificity_score(labels, predictions):
    """Compute specificity from binary labels and predictions."""
    tn, fp, _, _ = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    denominator = tn + fp

    return tn / denominator if denominator > 0 else np.nan


class TemperatureScaler(torch.nn.Module):
    """Learn a single temperature for post-hoc calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(
    logits,
    labels,
):
    """Learn the temperature using validation logits and labels."""

    scaler = TemperatureScaler()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.01,
        max_iter=50,
    )

    def closure():
        optimizer.zero_grad()

        scaled_logits = scaler(logits)

        loss = criterion(
            scaled_logits,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return scaler.temperature.item()
