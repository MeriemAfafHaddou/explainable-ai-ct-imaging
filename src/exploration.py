import pandas as pd


def get_slice_metadata(
    diagnosis_df: pd.DataFrame,
    patient_id: int,
    slice_number: int,
) -> pd.Series:
    """Get diagnosis metadata for a specific CT slice."""
    slice_metadata = diagnosis_df[
        (diagnosis_df["PatientNumber"] == patient_id)
        & (diagnosis_df["SliceNumber"] == slice_number)
    ]

    if slice_metadata.empty:
        raise ValueError(
            f"No metadata found for patient {patient_id}, slice {slice_number}"
        )

    if len(slice_metadata) > 1:
        raise ValueError(
            f"Multiple metadata entries found for patient {patient_id}, "
            f"slice {slice_number}"
        )

    return slice_metadata.iloc[0]


def describe_slice(
    diagnosis_df: pd.DataFrame,
    patient_id: int,
    slice_number: int,
) -> str:
    """Return a human-readable description of a CT slice."""
    metadata = get_slice_metadata(
        diagnosis_df,
        patient_id,
        slice_number,
    )

    if metadata["No_Hemorrhage"] == 1:
        description = "No intracranial hemorrhage is present."
    else:
        hemorrhage_types = [
            hemorrhage_type
            for hemorrhage_type in [
                "Intraventricular",
                "Intraparenchymal",
                "Subarachnoid",
                "Epidural",
                "Subdural",
            ]
            if metadata[hemorrhage_type] == 1
        ]

        if hemorrhage_types:
            types = ", ".join(
                hemorrhage_type.lower() for hemorrhage_type in hemorrhage_types
            )
            description = f"Intracranial hemorrhage is present ({types})."
        else:
            description = "Intracranial hemorrhage is present."

    if metadata["Fracture_Yes_No"] == 1:
        description += " A fracture is also present."

    return f"Patient {patient_id}, slice {slice_number}: {description}"


def get_example_slices(
    diagnosis_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Get one representative slice for each classification class."""
    return {
        "ICH": diagnosis_df[diagnosis_df["ICH"] == 1].iloc[0],
        "Non-ICH": diagnosis_df[diagnosis_df["ICH"] == 0].iloc[0],
    }
