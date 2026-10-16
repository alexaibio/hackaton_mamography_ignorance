import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table
from mammography.model.evaluate import (
    confusion,
    lift_table,
    metrics_at_threshold,
    threshold_free_metrics,
)
from mammography.model.train import (
    HORIZON_FLAGS,
    feature_columns,
    feature_importance,
    load_run,
    rebuild_test_pool,
    run_paths,
    save_run,
    train_model,
)

IMPORTANCE_METHODS = ("PredictionValuesChange", "LossFunctionChange")
DEFAULT_OFF = ("months_since_last_screen", "invitation_sequence", "prior_invitation_count")

st.set_page_config(page_title="Risk model", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


config = get_config()
categorical, numeric, boolean = feature_columns(config)
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
    st.warning(f"No saved model in {run_paths()['model'].parent}.")
else:
    st.info(
        f"**Saved model** — label `{existing['target']}`, "
        f"{len(existing['features'])} features, "
        f"{existing['iterations']} iterations (best: {existing['best_iteration']}), "
        f"tested on {existing['n_test']:,} invitations. "
        f"Trained at {existing.get('trained_at', 'this session')}.",
        icon=":material/save:",
    )

with st.form("training"):
    st.header("Features")
    st.caption(
        "Untick a feature to train without it. Edit `training_features` in config.yaml "
        "to change the available set."
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
                column, value=column not in DEFAULT_OFF, key=f"feature_{column}"
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

    train_requested = st.form_submit_button(
        "Re-train the model" if existing is not None else "Train the model", type="primary"
    )

if train_requested:
    with st.spinner(f"Training CatBoost on {len(selected)} features..."):
        run = train_model(
            invitations,
            target,
            iterations=int(iterations),
            test_size=test_size,
            selected=selected,
        )
        save_run(run)
        st.session_state["run"] = run
    st.toast(f"Saved to {run_paths()['model']}")

run = st.session_state.get("run")

if run is None:
    st.info("No model trained yet.")
    st.stop()

screened = run["y_test"]
probability = run["screening_probability"]

st.success(
    f"Trained on {run['n_train']:,} invitations, validated on {run['n_valid']:,}, "
    f"tested on {run['n_test']:,} ({run['persons_test']:,} patients). "
    f"Best iteration: {run['best_iteration']}. Label: `{run['target']}`. "
    f"Features: {len(run['features'])}."
)

st.header("Performance on the held-out test set")

st.subheader("Threshold-free")
st.table(pd.Series(threshold_free_metrics(screened, probability)).round(4).to_frame("value"))

with st.form("threshold"):
    threshold = st.slider(
        "Decision threshold on the probability of screening", 0.05, 0.95, 0.5, 0.05
    )
    st.form_submit_button("Apply")

threshold_left, threshold_right = st.columns(2)
with threshold_left:
    st.subheader("At the chosen threshold")
    st.table(
        pd.Series(metrics_at_threshold(screened, probability, threshold)).round(4).to_frame("value")
    )
with threshold_right:
    st.subheader("Confusion matrix")
    st.table(confusion(screened, probability, threshold))

st.subheader("Targeting value")
st.caption(
    "If outreach is limited to the highest-risk share of the cohort, how many "
    "non-responders does that reach compared with contacting at random."
)
st.table(lift_table(screened, probability))

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
