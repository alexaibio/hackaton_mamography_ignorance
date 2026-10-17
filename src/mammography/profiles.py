import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mammography.config import get_config
from mammography.model.train import feature_columns, feature_importance

POPULATION_LABEL = "all patients"


def feature_groups(selected: list[str] | None = None) -> dict[str, list[str]]:
    categorical, numeric, boolean = feature_columns(get_config(), selected)
    return {"categorical": categorical, "numeric": numeric, "boolean": boolean}


def most_important(run: dict, limit: int | None = None) -> list[str]:
    """The model's own features, ranked by importance."""
    ranked = feature_importance(run["model"], None, "PredictionValuesChange")
    return ranked.index.tolist()[:limit]


def person_level(invitations: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per person: her invitation history plus her latest attributes."""
    ordered = invitations.sort_values(["person_id", "invitation_date"])

    history = ordered.groupby("person_id").agg(
        invitations=("invitation_id", "size"),
        screenings=(target, "sum"),
        # not first_invitation / last_invitation: first_invitation is a feature column
        first_invited_at=("invitation_date", "min"),
        last_invited_at=("invitation_date", "max"),
    )
    history["response_rate"] = history["screenings"] / history["invitations"]

    latest = ordered.groupby("person_id").tail(1).set_index("person_id")
    return history.join(latest.drop(columns=["invitation_id"]))


def profile(people: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """The typical member of a group, one line per feature."""
    rows = []
    for column in groups["numeric"]:
        rows.append((column, "median", f"{people[column].median():g}"))
    for column in groups["boolean"]:
        rows.append((column, "share true", f"{people[column].mean():.1%}"))
    for column in groups["categorical"]:
        shares = people[column].value_counts(normalize=True, dropna=False)
        rows.append((column, "most common", f"{shares.index[0]} ({shares.iloc[0]:.0%})"))
    return pd.DataFrame(rows, columns=["feature", "statistic", "typical value"]).set_index(
        "feature"
    )


def contrast(
    group: pd.DataFrame, others: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    """Which numeric and boolean features separate a group from everyone else."""
    rows = []

    for column in groups["numeric"]:
        here, there = group[column].median(), others[column].median()
        spread = pd.concat([group[column], others[column]]).std()
        rows.append(
            {
                "feature": column,
                "kind": "median",
                "this group": round(float(here), 2),
                "the others": round(float(there), 2),
                "difference": round(float(here - there), 2),
                "effect": round(float((here - there) / spread), 3) if spread else 0.0,
            }
        )

    for column in groups["boolean"]:
        here, there = group[column].mean(), others[column].mean()
        rows.append(
            {
                "feature": column,
                "kind": "share true",
                "this group": round(float(here), 4),
                "the others": round(float(there), 4),
                "difference": round(float(here - there), 4),
                "effect": round(float(here - there), 3),
            }
        )

    table = pd.DataFrame(rows)
    return table.reindex(table["effect"].abs().sort_values(ascending=False).index).set_index(
        "feature"
    )


def category_gaps(
    group: pd.DataFrame, others: pd.DataFrame, column: str, top: int = 8
) -> pd.DataFrame:
    here = group[column].value_counts(normalize=True, dropna=False)
    there = others[column].value_counts(normalize=True, dropna=False)

    table = pd.DataFrame({"this group": here, "the others": there}).fillna(0.0)
    table["gap_pp"] = ((table["this group"] - table["the others"]) * 100).round(1)
    table["this group"] = (table["this group"] * 100).round(1)
    table["the others"] = (table["the others"] * 100).round(1)

    return table.reindex(table["gap_pp"].abs().sort_values(ascending=False).index).head(top)


def normalised_groups(
    frames: dict[str, pd.DataFrame], population: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    """Numeric and boolean features on a common 0-1 scale, one column per group.

    Counts are divided by the population's 90th percentile, so a typical high value
    reaches about 1 and the axes stay comparable. Shares are already 0-1.
    """
    rows = {}

    for column in groups["numeric"]:
        scale = population[column].quantile(0.9)
        if not scale or pd.isna(scale):
            continue
        rows[column] = {
            label: frame[column].median() / scale for label, frame in frames.items()
        }

    for column in groups["boolean"]:
        rows[column] = {label: frame[column].mean() for label, frame in frames.items()}

    return pd.DataFrame(rows).T.clip(lower=0.0, upper=1.2).fillna(0.0)


def normalised_profile(
    group: pd.DataFrame,
    population: pd.DataFrame,
    groups: dict[str, list[str]],
    group_label: str,
) -> pd.DataFrame:
    return normalised_groups(
        {group_label: group, POPULATION_LABEL: population}, population, groups
    )


def radar_figure(shape: pd.DataFrame, colours: dict[str, str]):
    """The normalised profile as a radar. Series are drawn in the order given."""
    labels = [
        name.replace("_cnt", "").replace("_flag", "").replace("_", " ") for name in shape.index
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed = angles + angles[:1]

    figure, axes = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for series, colour in colours.items():
        values = shape[series].tolist()
        axes.plot(closed, values + values[:1], color=colour, linewidth=2, label=series)
        axes.fill(closed, values + values[:1], color=colour, alpha=0.15)

    axes.set_xticks(angles)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_yticks([0.25, 0.5, 0.75, 1.0])
    axes.set_yticklabels([])
    axes.set_ylim(0, 1.2)
    axes.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12))
    return figure


def representative(people: pd.DataFrame, groups: dict[str, list[str]]) -> pd.Series:
    """The real patient closest to the group's own median profile."""
    columns = groups["numeric"] + groups["boolean"]
    values = people[columns].astype(float).fillna(0.0)

    spread = values.std().replace(0, 1).fillna(1)
    distance = ((values - values.median()) / spread).abs().sum(axis=1)

    return people.loc[distance.idxmin()]
