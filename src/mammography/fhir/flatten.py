import json
from pathlib import Path

import pandas as pd

from mammography.config import get_settings
from mammography.io import write_table


def read_ndjson(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten_patients(resources: list[dict]) -> pd.DataFrame:
    rows = []
    for resource in resources:
        address = (resource.get("address") or [{}])[0]
        rows.append(
            {
                "patient_id": resource["id"],
                "birth_date": resource.get("birthDate"),
                "gender": resource.get("gender"),
                "deceased_date": resource.get("deceasedDateTime"),
                "city": address.get("city"),
                "district": address.get("district"),
                "state": address.get("state"),
                "postal_code": address.get("postalCode"),
                "country": address.get("country"),
            }
        )
    return pd.DataFrame(rows)


def flatten_resources() -> None:
    settings = get_settings()
    patients = flatten_patients(read_ndjson(settings.raw_dir / "Patient.ndjson"))
    write_table(patients, "interim", "patients")
