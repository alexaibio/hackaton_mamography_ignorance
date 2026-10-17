import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table
from mammography.model.evaluate import ignore_class_metrics, operating_point, roc_frame
from mammography.model.train import (
    HORIZON_FLAGS,
    feature_columns,
    feature_importance,
    load_run,
    models_folder,
    rebuild_test_pool,
    run_paths,
    save_run,
    saved_targets,
    train_model,
)

IMPORTANCE_METHODS = ("PredictionValuesChange", "LossFunctionChange")
EXTREME_FEATURES = 7

# redundant with a feature that is kept, so off by default:
#   invitation_sequence        — the same number as prior_invitation_count
#   months_since_last_screen   — null exactly when prior_screen_count_5y is 0
#   never_screened_before_flag — true exactly when prior_screen_count_5y is 0
DEFAULT_OFF = (
    "invitation_sequence",
    "months_since_last_screen",
    "never_screened_before_flag",
)

st.set_page_config(page_title="Risk model", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


config = get_config()
categorical, numeric, boolean = feature_columns(config)
ALL_FEATURES = categorical + numeric + boolean
invitations = load_invitations()

st.title("Risk model")
st.caption(
    "CatBoost gradient-boosted trees. Predicts whether a screening is billed within the "
    "chosen horizon; risk of ignoring is 1 minus that probability."
)

if "run" not in st.session_state:
    saved = load_run()
    if saved is not None:
        st.session_state["run"] = saved

existing = st.session_state.get("run")

if existing is None:
    st.warning(f"No saved model in {models_folder()}.")
    trained_on = {column for column in ALL_FEATURES if column not in DEFAULT_OFF}
else:
    trained_on = set(existing["features"])

with st.form("training"):
    st.header("Features")
    st.caption(
        "Ticked boxes start from the features the saved model was actually trained on. "
        "Edit `training_features` in config.yaml to change the available set."
    )

    selected = []
    for group_name, group in [
        ("categorical", categorical),
        ("numeric", numeric),
        ("boolean", boolean),
    ]:
        st.write(f"**{group_name}**")
        checkboxes = st.columns(4)
        for position, column in enumerate(group):
            if checkboxes[position % 4].checkbox(
                column, value=column in trained_on, key=f"feature_{column}"
            ):
                selected.append(column)

    st.header("Label and training")

    label_column, size_column, iterations_column = st.columns(3)

    target = label_column.selectbox(
        "Label",
        list(HORIZON_FLAGS)[::-1],
        index=0,
        help="the other horizons are the same outcome earlier or later, never used as features",
    )
    test_size = size_column.slider("Test set share", 0.1, 0.4, config["split"]["test_size"], 0.05)
    iterations = iterations_column.number_input(
        "Iterations",
        min_value=50,
        max_value=5000,
        value=int(config["catboost"]["iterations"]),
        step=50,
    )

    st.caption(
        "Train/validation/test are split by `person_id`, so no woman appears on both sides — "
        "a random row split would leak her behaviour across the boundary."
    )

    all_horizons = st.checkbox(
        "Also train the other two horizons",
        value=False,
        help=(
            "needed for the response-speed profile on the Patient risk page; triples the "
            "training time, so lower the iterations first"
        ),
    )

    train_requested = st.form_submit_button(
        "Re-train the model" if existing is not None else "Train the model", type="primary"
    )

if train_requested:
    horizons = [target]
    if all_horizons:
        horizons += [flag for flag in HORIZON_FLAGS if flag != target]

    for horizon in horizons:
        with st.spinner(f"Training `{horizon}` on {len(selected)} features..."):
            trained = train_model(
                invitations,
                horizon,
                iterations=int(iterations),
                test_size=test_size,
                selected=selected,
            )
            save_run(trained)
            if horizon == target:
                st.session_state["run"] = trained
                trained_on = set(trained["features"])
        st.toast(f"Saved {run_paths(horizon)['model'].name}")

run = st.session_state.get("run")

if run is None:
    st.info("No model trained yet.")
    st.stop()

screened = run["y_test"]
probability = run["screening_probability"]

st.success(
    f"**Model in use** — label `{run['target']}`, {len(run['features'])} features, "
    f"{run['iterations']} iterations (best: {run['best_iteration']}). "
    f"Trained on {run['n_train']:,} invitations, validated on {run['n_valid']:,}, "
    f"tested on {run['n_test']:,} ({run['persons_test']:,} patients). "
    f"Trained at {run.get('trained_at', 'this session')}."
)

available = saved_targets()
st.caption(
    "Horizons with a saved model: "
    + ", ".join(f"`{horizon}`" for horizon in available)
    + (
        ""
        if len(available) == len(HORIZON_FLAGS)
        else " — tick *Also train the other two horizons* to enable the response-speed "
        "profile on the Patient risk page."
    )
)

with st.expander(f"The {len(run['features'])} features this model was trained on"):
    st.table(
        pd.DataFrame(
            [
                (
                    column,
                    "categorical"
                    if column in run["categorical"]
                    else ("boolean" if column in boolean else "numeric"),
                )
                for column in run["features"]
            ],
            columns=["feature", "type"],
        ).set_index("feature")
    )

unticked = trained_on - set(selected)
newly_ticked = set(selected) - trained_on
if unticked or newly_ticked:
    st.warning(
        "The tickboxes no longer match this model. Everything below still describes the "
        "model as trained — press **Re-train the model** to apply the change.\n\n"
        + (f"- removed since training: `{'`, `'.join(sorted(unticked))}`\n" if unticked else "")
        + (f"- added since training: `{'`, `'.join(sorted(newly_ticked))}`" if newly_ticked else "")
    )

st.header("Most important features")

method = st.radio("Method", IMPORTANCE_METHODS, horizontal=True)

pool = run.get("test_pool")
if method == "LossFunctionChange" and pool is None:
    with st.spinner("Rebuilding the test set for the loss-based importance..."):
        pool = rebuild_test_pool(invitations, run)
        run["test_pool"] = pool

importance = feature_importance(run["model"], pool, method)

importance_left, importance_right = st.columns([2, 3])
importance_left.table(importance.round(3))
importance_right.bar_chart(importance)

st.subheader("The extremes")

strongest, weakest = st.columns(2)
with strongest:
    st.markdown(f"#### {EXTREME_FEATURES} most important — what the model leans on")
    st.table(importance.head(EXTREME_FEATURES).round(3))
with weakest:
    st.markdown(
        f"#### {EXTREME_FEATURES} least important — candidates to untick and retrain without"
    )
    st.table(importance.tail(EXTREME_FEATURES).round(3))

if len(importance) < 2 * EXTREME_FEATURES:
    st.caption(
        f"The model has {len(importance)} features, so the two lists overlap — some "
        "features appear in both."
    )

st.header("Performance on the held-out test set")

st.caption(
    "An invitation is flagged when the predicted probability of screening falls below "
    f"the threshold. Non-response is {1 - screened.mean():.1%} of the test set, so "
    "flagging everything would already give that precision — the model has to beat it."
)

if "threshold_value" not in st.session_state:
    st.session_state["threshold_value"] = 0.5


def remember_threshold() -> None:
    # a plain key survives leaving the page; a widget key is garbage collected
    st.session_state["threshold_value"] = st.session_state["threshold_slider"]


threshold = st.slider(
    "Decision threshold on the probability of screening",
    0.01,
    0.99,
    value=st.session_state["threshold_value"],
    step=0.01,
    key="threshold_slider",
    on_change=remember_threshold,
    help="applies immediately — the table and the marker on the curve both follow it",
)

curve, auc = roc_frame(screened, probability)
false_positive_rate, recall = operating_point(screened, probability, threshold)
flagged_share = float((probability < threshold).mean())

metrics_column, curve_column = st.columns([2, 3])

with metrics_column:
    st.subheader("At this threshold")
    st.table(
        pd.Series(ignore_class_metrics(screened, probability, threshold))
        .round(4)
        .to_frame("value")
    )
    st.metric("Share of the cohort flagged", f"{flagged_share:.1%}")
    st.caption(
        "Flagging almost everyone gets high recall for free. The number worth watching "
        "is precision against the base rate above."
    )

with curve_column:
    st.subheader("ROC curve")
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot(curve.index, curve["model"], label=f"model (AUC {auc:.3f})")
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="random")
    axes.scatter(
        [false_positive_rate],
        [recall],
        color="crimson",
        zorder=5,
        label=f"threshold {threshold:.2f}",
    )
    axes.axvline(false_positive_rate, color="crimson", linewidth=0.5, linestyle=":")
    axes.axhline(recall, color="crimson", linewidth=0.5, linestyle=":")
    axes.set_xlabel("false positive rate — women who do come, wrongly flagged")
    axes.set_ylabel("true positive rate — non-responders caught (recall)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(loc="lower right")
    st.pyplot(figure)

    st.caption(
        f"The curve itself does not move with the threshold — it is drawn across every "
        f"threshold, which is what makes **AUC {auc:.4f}** a judgement of the model "
        "rather than of one operating point. The red dot is where this threshold puts "
        "you. 0.65-0.75 is a genuinely useful AUC for administrative data."
    )
