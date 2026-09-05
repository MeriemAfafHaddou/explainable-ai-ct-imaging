from pathlib import Path

import pandas as pd
from PIL import Image


def load_diagnosis(data_dir: Path) -> pd.DataFrame:
    """Load slice-level hemorrhage diagnosis metadata."""
    diagnosis_path = data_dir / "hemorrhage_diagnosis.csv"

    if not diagnosis_path.exists():
        raise FileNotFoundError(f"Diagnosis file not found: {diagnosis_path}")

    return pd.read_csv(diagnosis_path)


def load_demographics(data_dir: Path) -> pd.DataFrame:
    """Load patient-level demographic metadata."""
    demographics_path = data_dir / "patient_demographics.csv"

    if not demographics_path.exists():
        raise FileNotFoundError(f"Demographics file not found: {demographics_path}")

    return pd.read_csv(demographics_path)


def load_ct_image(
    data_dir: Path,
    patient_id: int,
    slice_number: int,
    window: str = "brain",
) -> Image.Image:
    """Load a CT slice for a given patient and imaging window."""
    if window not in {"brain", "bone"}:
        raise ValueError("window must be either 'brain' or 'bone'")

    image_path = (
        data_dir / "Patients_CT" / f"{patient_id:03d}" / window / f"{slice_number}.jpg"
    )

    if not image_path.exists():
        raise FileNotFoundError(f"CT image not found: {image_path}")

    return Image.open(image_path)
