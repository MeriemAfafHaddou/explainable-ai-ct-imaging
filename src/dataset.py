import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_loading import load_ct_image


class CTICHImageDataset(Dataset):
    """PyTorch dataset for brain-window CT slice classification."""

    def __init__(
        self,
        diagnosis_df: pd.DataFrame,
        data_dir,
        patient_ids,
        transform=None,
    ):
        self.data_dir = data_dir
        self.transform = transform

        self.samples = diagnosis_df[
            diagnosis_df["PatientNumber"].isin(patient_ids)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]

        patient_id = int(sample["PatientNumber"])
        slice_number = int(sample["SliceNumber"])
        label = int(sample["ICH"])

        image = load_ct_image(
            self.data_dir,
            patient_id=patient_id,
            slice_number=slice_number,
            window="brain",
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class CTDiffusionDataset(Dataset):
    def __init__(self, diagnosis_df, data_dir, transform=None):
        self.diagnosis_df = diagnosis_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.diagnosis_df)

    def __getitem__(self, idx):
        row = self.diagnosis_df.iloc[idx]

        patient_id = int(row["PatientNumber"])
        slice_number = int(row["SliceNumber"])

        image = load_ct_image(
            self.data_dir,
            patient_id=patient_id,
            slice_number=slice_number,
            window="brain",
        )

        if self.transform:
            image = self.transform(image)

        return image
