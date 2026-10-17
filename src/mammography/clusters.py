import matplotlib.pyplot as plt
import pandas as pd
from catboost import Pool
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mammography.config import get_config
from mammography.llm.messages import FEATURE_LABELS
from mammography.model.train import MISSING, feature_columns


def model_frame(people: pd.DataFrame, run: dict) -> pd.DataFrame:
    """The model's features, typed the way training typed them."""
    _, _, boolean = feature_columns(get_config())
    frame = people[run["features"]].copy()

    for column in run["features"]:
        if column in run["categorical"]:
            frame[column] = frame[column].fillna(MISSING).astype(str)
        elif column in boolean:
            frame[column] = frame[column].astype("int8")
        else:
            frame[column] = pd.to_numeric(frame[column])

    return frame


def shap_clusters(
    run: dict, people: pd.DataFrame, clusters: int = 4, sample: int = 20_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group patients by why the model thinks they are at risk, not by raw similarity.

    Returns the sampled patients with their cluster, risk and 2D coordinates, and the
    per-patient SHAP contributions behind them.
    """
    seed = get_config()["seed"]
    taken = people.sample(min(sample, len(people)), random_state=seed)

    frame = model_frame(taken, run)
    pool = Pool(frame, cat_features=run["categorical"])

    contributions = pd.DataFrame(
        run["model"].get_feature_importance(type="ShapValues", data=pool)[:, :-1],
        columns=run["features"],
        index=taken.index,
    )
    risk = 1 - run["model"].predict_proba(pool)[:, 1]

    labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(contributions)
    coordinates = PCA(n_components=2, random_state=seed).fit_transform(contributions)

    points = pd.DataFrame(
        {
            "cluster": labels,
            "risk": risk,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        },
        index=taken.index,
    )
    return points, contributions


def describe_clusters(
    points: pd.DataFrame, contributions: pd.DataFrame, drivers: int = 3
) -> pd.DataFrame:
    """One row per cluster: how big, how risky, and what drives it."""
    rows = []

    for label, group in points.groupby("cluster"):
        average = contributions.loc[group.index].mean()
        strongest = average.reindex(average.abs().sort_values(ascending=False).index).head(drivers)
        rows.append(
            {
                "cluster": int(label),
                "patients": len(group),
                "share": round(len(group) / len(points), 3),
                "mean risk": round(float(group["risk"].mean()), 3),
                "main drivers": ", ".join(
                    f"{name} ({value:+.2f})" for name, value in strongest.items()
                ),
            }
        )

    return pd.DataFrame(rows).set_index("cluster")


def typical_value(people: pd.DataFrame, column: str, index) -> str:
    values = people.loc[index, column]

    if values.dtype == bool:
        return "yes" if values.mean() >= 0.5 else "no"
    if pd.api.types.is_numeric_dtype(values):
        return f"{values.median():g}"
    return str(values.mode().iloc[0])


def cluster_descriptions(
    points: pd.DataFrame,
    contributions: pd.DataFrame,
    people: pd.DataFrame,
    drivers: int = 3,
) -> list[str]:
    """One readable paragraph per cluster, built from its strongest SHAP drivers."""
    descriptions = []

    for label, group in points.groupby("cluster"):
        average = contributions.loc[group.index].mean()
        strongest = average.reindex(average.abs().sort_values(ascending=False).index).head(drivers)

        against, towards = [], []
        for name, value in strongest.items():
            phrase = (
                f"{FEATURE_LABELS.get(name, name.replace('_', ' '))} "
                f"(typically **{typical_value(people, name, group.index)}**)"
            )
            (against if value < 0 else towards).append(phrase)

        blocks = [
            f"**Group {int(label)}** — {len(group):,} patients, "
            f"{len(group) / len(points):.0%} of the sample, "
            f"average risk of refusal **{group['risk'].mean():.0%}**"
        ]
        if against:
            blocks.append(
                "What keeps them away:\n" + "\n".join(f"- {phrase}" for phrase in against)
            )
        if towards:
            blocks.append(
                "What works in their favour:\n" + "\n".join(f"- {phrase}" for phrase in towards)
            )

        descriptions.append("\n\n".join(blocks))

    return descriptions


def cluster_figure(points: pd.DataFrame):
    """The sampled patients in SHAP space, coloured by cluster."""
    figure, axes = plt.subplots(figsize=(6, 5))

    for label, group in points.groupby("cluster"):
        axes.scatter(
            group["x"],
            group["y"],
            s=6,
            alpha=0.35,
            label=f"cluster {int(label)} ({len(group):,})",
        )

    axes.set_xlabel("SHAP component 1")
    axes.set_ylabel("SHAP component 2")
    axes.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    return figure
