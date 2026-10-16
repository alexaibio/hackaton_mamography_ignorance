import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit

from mammography.config import get_config

HORIZON_FLAGS = ("screening_3m_flag", "screening_6m_flag", "screening_12m_flag")
MISSING = "missing"


def feature_columns(
    config: dict, selected: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    features = config["training_features"]
    groups = (
        list(features["categorical"]),
        list(features["numeric"]),
        list(features.get("boolean", [])),
    )
    if selected is None:
        return groups

    keep = set(selected)
    return tuple([column for column in group if column in keep] for group in groups)


def prepare(
    invitations: pd.DataFrame, target: str, config: dict, selected: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    categorical, numeric, boolean = feature_columns(config, selected)
    columns = categorical + numeric + boolean

    if not columns:
        raise ValueError("no features selected")

    leaking = set(columns) & set(HORIZON_FLAGS)
    if leaking:
        raise ValueError(f"{sorted(leaking)} is the label at another horizon, not a feature")

    missing = [column for column in columns if column not in invitations.columns]
    if missing:
        raise KeyError(f"columns missing from the invitations table: {missing}")

    features = invitations[columns].copy()
    for column in categorical:
        features[column] = features[column].fillna(MISSING).astype(str)
    for column in boolean:
        features[column] = features[column].astype("int8")

    return features, invitations[target].astype(int)


def split_by_person(
    features: pd.DataFrame, labels: pd.Series, persons: pd.Series, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(features, labels, groups=persons))


def train_model(
    invitations: pd.DataFrame,
    target: str,
    iterations: int | None = None,
    test_size: float | None = None,
    selected: list[str] | None = None,
) -> dict:
    config = get_config()
    seed = config["seed"]
    categorical, _, _ = feature_columns(config, selected)

    features, labels = prepare(invitations, target, config, selected)
    persons = invitations["person_id"]

    holdout = test_size if test_size is not None else config["split"]["test_size"]
    fit_index, test_index = split_by_person(features, labels, persons, holdout, seed)

    # a second split so early stopping never sees the test set
    inner_fit, inner_valid = split_by_person(
        features.iloc[fit_index],
        labels.iloc[fit_index],
        persons.iloc[fit_index],
        holdout,
        seed,
    )
    train_index = fit_index[inner_fit]
    valid_index = fit_index[inner_valid]

    parameters = dict(config["catboost"])
    if iterations is not None:
        parameters["iterations"] = iterations

    train_pool = Pool(
        features.iloc[train_index], labels.iloc[train_index], cat_features=categorical
    )
    valid_pool = Pool(
        features.iloc[valid_index], labels.iloc[valid_index], cat_features=categorical
    )
    test_pool = Pool(features.iloc[test_index], labels.iloc[test_index], cat_features=categorical)

    model = CatBoostClassifier(random_seed=seed, **parameters)
    model.fit(train_pool, eval_set=valid_pool)

    return {
        "model": model,
        "target": target,
        "features": list(features.columns),
        "categorical": categorical,
        "n_train": len(train_index),
        "n_valid": len(valid_index),
        "n_test": len(test_index),
        "persons_test": int(persons.iloc[test_index].nunique()),
        "best_iteration": model.get_best_iteration(),
        "y_test": labels.iloc[test_index].to_numpy(),
        "screening_probability": model.predict_proba(test_pool)[:, 1],
        "test_pool": test_pool,
    }


def feature_importance(model: CatBoostClassifier, pool: Pool, method: str) -> pd.DataFrame:
    values = model.get_feature_importance(data=pool if method != "PredictionValuesChange" else None,
                                          type=method)
    return (
        pd.DataFrame({"feature": model.feature_names_, "importance": values})
        .sort_values("importance", ascending=False)
        .set_index("feature")
    )
