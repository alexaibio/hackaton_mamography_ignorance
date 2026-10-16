from pathlib import Path

import pandas as pd

from mammography.config import get_settings
from mammography.io import write_table

DATE_COLUMNS = ("invitation_date",)

# czech A/N = ano/ne
FLAG_COLUMNS = (
    "first_invitation",
    "gp_registered",
    "gyn_registered_flag",
    "never_screened_before_flag",
    "screening_3m_flag",
    "screening_6m_flag",
    "screening_12m_flag",
)

ID_COLUMNS = ("INVITATION_ID", "PERSON_ID")


def normalise(invitations: pd.DataFrame) -> pd.DataFrame:
    invitations.columns = [column.lower() for column in invitations.columns]

    for column in DATE_COLUMNS:
        invitations[column] = pd.to_datetime(invitations[column], format="%d.%m.%Y")

    for column in FLAG_COLUMNS:
        invitations[column] = invitations[column].map({"A": True, "N": False})

    return invitations


def default_source() -> Path:
    directory = get_settings().raw_csv_dir
    candidates = sorted(directory.glob("*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one csv in {directory}, found {len(candidates)}: "
            f"{[path.name for path in candidates]} — pass --path to pick one"
        )
    return candidates[0]


def load_invitations_csv(path: Path, encoding: str = "utf-8") -> int:
    invitations = pd.read_csv(path, encoding=encoding, dtype={name: str for name in ID_COLUMNS})
    write_table(normalise(invitations), "interim", "invitations")
    return len(invitations)
