"""Read the patient JSON files from the supplied dataset folder."""
import json
from pathlib import Path
import ast


def parse_similar_patients(value):
    """Convert the dataset's string representation of a dictionary to a dict."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("similar_patients must contain a dictionary")
    return parsed


def load_dataset(input_folder):
    """Load and validate all patient JSON files in a directory recursively."""
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    records = []
    for path in sorted(folder.rglob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        required = ("patient_uid", "patient_id", "patient")
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        record["_similar_patients"] = parse_similar_patients(
            record.get("similar_patients", "{}")
        )
        records.append(record)

    if not records:
        raise ValueError(f"No JSON patient files found in {folder}")

    uids = [record["patient_uid"] for record in records]
    if len(uids) != len(set(uids)):
        raise ValueError("Duplicate patient_uid values found in the dataset")
    return records
