import pandas as pd
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


def get_train_transforms():
    """Return preprocessing and augmentation transforms for training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
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
