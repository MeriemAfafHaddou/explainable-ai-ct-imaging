import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.data_loading import load_ct_image


class CTICHAutoencoderDataset(Dataset):
    """Dataset for reconstructing brain-window CT slices."""

    def __init__(
        self,
        diagnosis_df,
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

        image = load_ct_image(
            self.data_dir,
            patient_id=patient_id,
            slice_number=slice_number,
            window="brain",
        )

        if self.transform is not None:
            image = self.transform(image)

        return image


def get_autoencoder_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


class CTICHAutoencoder(nn.Module):
    """Small convolutional autoencoder for brain CT slices."""

    def __init__(self, latent_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 28 * 28),
            nn.ReLU(),

            nn.Unflatten(1, (128, 28, 28)),

            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)

        return reconstruction


def prepare_for_classifier(image):
    """Convert autoencoder output to classifier input."""

    image = image.repeat(1, 3, 1, 1)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image.device,
    ).view(1, 3, 1, 1)

    image = (image - mean) / std

    return image
