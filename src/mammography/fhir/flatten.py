import json
from pathlib import Path

import pandas as pd

from mammography.config import get_settings
from mammography.io import write_table


def read_ndjson(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def first_coding_code(concept: dict | None) -> str | None:
    coding = (concept or {}).get("coding") or [{}]
    return coding[0].get("code")


def preferred_language(resource: dict) -> str | None:
    communication = resource.get("communication") or []
    if not communication:
        return None
    return first_coding_code(communication[0].get("language"))


def flatten_patients(resources: list[dict]) -> pd.DataFrame:
    rows = []
    for resource in resources:
        address = (resource.get("address") or [{}])[0]
        telecom = resource.get("telecom") or []
        systems = {entry.get("system") for entry in telecom}
        uses = {entry.get("use") for entry in telecom}

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
                "marital_status": first_coding_code(resource.get("maritalStatus")),
                "preferred_language": preferred_language(resource),
                "has_phone": "phone" in systems,
                "has_mobile": "mobile" in uses or "sms" in systems,
                "has_email": "email" in systems,
                "has_registered_gp": bool(resource.get("generalPractitioner")),
            }
        )
    return pd.DataFrame(rows)


def flatten_resources() -> None:
    settings = get_settings()
    patients = flatten_patients(read_ndjson(settings.raw_dir / "Patient.ndjson"))
    write_table(patients, "interim", "patients")
