import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss


def compute_brier_score(labels, probabilities):
    """Compute the Brier score for binary classification."""
    return brier_score_loss(labels, probabilities)


def compute_ece(labels, probabilities, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probabilities >= lower) & (probabilities < upper)

        if not np.any(in_bin):
            continue

        bin_confidence = probabilities[in_bin].mean()
        bin_accuracy = labels[in_bin].mean()
        bin_fraction = in_bin.mean()

        ece += bin_fraction * abs(bin_accuracy - bin_confidence)

    return ece


def fit_temperature(logits, labels):
    """Fit a temperature parameter using validation logits."""
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    def objective(log_temperature):
        temperature = torch.exp(
            torch.tensor(log_temperature, dtype=torch.float32)
        )

        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels)

        return loss.item()

    result = minimize_scalar(
        objective,
        bounds=(-2.0, 2.0),
        method="bounded",
    )

    temperature = np.exp(result.x)

    return float(temperature)
