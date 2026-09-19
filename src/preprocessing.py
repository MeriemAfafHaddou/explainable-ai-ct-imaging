import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_patient_labels(
    diagnosis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one binary ICH label per patient."""
    patient_labels = diagnosis_df.groupby("PatientNumber")["ICH"].max().reset_index()

    patient_labels["ICH"] = patient_labels["ICH"].astype(int)

    return patient_labels


def split_patients(
    patient_labels: pd.DataFrame,
    test_size: int = 17,
    random_state: int = 42,
):
    """Split patients into development and held-out test sets."""
    development_patients, test_patients = train_test_split(
        patient_labels,
        test_size=test_size,
        stratify=patient_labels["ICH"],
        random_state=random_state,
    )

    return development_patients, test_patients


def create_stratified_folds(
    development_patients: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
):
    """Create stratified patient-level cross-validation folds."""
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(
            development_patients,
            development_patients["ICH"],
        ),
        start=1,
    ):
        folds.append(
            {
                "fold": fold,
                "train": development_patients.iloc[train_idx],
                "validation": development_patients.iloc[val_idx],
            }
        )

    return folds


class AddGaussianNoise:
    """Add mild additive Gaussian noise to emulate scanner acquisition noise.

    Applied AFTER ToTensor() (i.e. on the normalized tensor), so `std` is in
    normalized-intensity units, not raw pixel/HU units. Kept small and
    zero-mean so it doesn't shift the mean intensity that carries the
    hemorrhage (hyperdensity) signal -- it only adds realistic per-pixel
    noise variation around it.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_train_transforms():
    """Return preprocessing and CT-specific training augmentations."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
            AddGaussianNoise(mean=0.0, std=0.02),
        ]
    )


def get_eval_transforms():
    """Return preprocessing transforms for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def compute_class_weights(labels, num_classes=2):
    """Compute inverse-frequency class weights for a classification loss."""
    labels = np.asarray(labels)

    counts = np.bincount(labels, minlength=num_classes)

    if np.any(counts == 0):
        raise ValueError("All classes must have at least one sample.")

    weights = len(labels) / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)
