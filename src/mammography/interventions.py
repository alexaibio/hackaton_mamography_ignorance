from pathlib import Path

import pandas as pd

from mammography.config import get_settings

TEXTS_FILE = "proposed_correction.csv"


def texts_path() -> Path | None:
    """Next to the package, falling back to the data folder."""
    candidates = (
        Path(__file__).with_name(TEXTS_FILE),
        get_settings().data_folder / TEXTS_FILE,
    )
    return next((path for path in candidates if path.exists()), None)


def proposed_interventions() -> dict[str, str]:
    """feature name -> the text to show for it. Empty when the file is absent."""
    path = texts_path()
    if path is None:
        return {}

    table = pd.read_csv(path)
    feature_column, text_column = table.columns[:2]
    return dict(zip(table[feature_column].astype(str), table[text_column].astype(str)))
