import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_free_metrics(screened: np.ndarray, screening_probability: np.ndarray) -> dict:
    return {
        "ROC AUC": roc_auc_score(screened, screening_probability),
        "Average precision": average_precision_score(screened, screening_probability),
        "Brier score": brier_score_loss(screened, screening_probability),
    }


def metrics_at_threshold(
    screened: np.ndarray, screening_probability: np.ndarray, threshold: float
) -> dict:
    predicted = (screening_probability >= threshold).astype(int)
    return {
        "Precision": precision_score(screened, predicted, zero_division=0),
        "Recall": recall_score(screened, predicted, zero_division=0),
        "F1": f1_score(screened, predicted, zero_division=0),
        "Flagged as responders": predicted.mean(),
    }


def ignore_class_metrics(
    screened: np.ndarray, screening_probability: np.ndarray, threshold: float
) -> dict:
    """Metrics for the class that is acted on: flagged as likely to ignore."""
    ignored = (1 - screened).astype(int)
    flagged = (screening_probability < threshold).astype(int)
    return {
        "Precision — of those flagged, really ignore": precision_score(
            ignored, flagged, zero_division=0
        ),
        "Recall — of all who ignore, flagged": recall_score(ignored, flagged, zero_division=0),
    }


def capacity_outcome(
    screened: np.ndarray, screening_probability: np.ndarray, capacity_share: float
) -> dict:
    """What contacting the highest-risk capacity_share of the cohort would reach."""
    ignored = 1 - screened
    risk = 1 - screening_probability
    size = max(int(len(risk) * capacity_share), 1)

    highest_risk = np.argsort(-risk)[:size]
    reached = int(ignored[highest_risk].sum())
    at_random = int(round(size * ignored.mean()))

    return {
        "contacted": size,
        "reached": reached,
        "at_random": at_random,
        "extra": reached - at_random,
        "precision": reached / size,
        "share_of_all_non_responders": reached / ignored.sum(),
        "risk_cutoff": float(risk[highest_risk][-1]),
    }


def confusion(
    screened: np.ndarray, screening_probability: np.ndarray, threshold: float
) -> pd.DataFrame:
    predicted = (screening_probability >= threshold).astype(int)
    matrix = confusion_matrix(screened, predicted, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["actually ignored", "actually screened"],
        columns=["predicted ignore", "predicted screening"],
    )


def lift_table(
    screened: np.ndarray, screening_probability: np.ndarray, fractions=(0.05, 0.1, 0.2, 0.3, 0.5)
) -> pd.DataFrame:
    ignored = 1 - screened
    risk = 1 - screening_probability
    ranked = ignored[np.argsort(-risk)]
    base_rate = ignored.mean()

    rows = []
    for fraction in fractions:
        size = max(int(len(risk) * fraction), 1)
        captured = ranked[:size].mean()
        rows.append(
            {
                "top % by risk": f"{fraction:.0%}",
                "non_response_rate": round(captured, 4),
                "lift vs random": round(captured / base_rate, 2),
                "share of all non-responders found": round(
                    ranked[:size].sum() / ignored.sum(), 4
                ),
            }
        )
    return pd.DataFrame(rows).set_index("top % by risk")
