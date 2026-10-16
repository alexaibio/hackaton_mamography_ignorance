import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table
from mammography.model.evaluate import ignore_class_metrics
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
    st.warning(f"No saved model in {run_paths()['model'].parent}.")
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
        trained_on = set(run["features"])
    st.toast(f"Saved to {run_paths()['model']}")

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

st.header("Performance on the held-out test set")

with st.form("threshold"):
    threshold = st.slider(
        "Decision threshold on the probability of screening", 0.01, 0.99, 0.5, 0.01
    )
    st.form_submit_button("Apply")

st.subheader("The class we act on — flagged as likely to ignore")
st.caption(
    "An invitation is flagged when the predicted probability of screening falls below "
    "the threshold."
)

st.table(
    pd.Series(ignore_class_metrics(screened, probability, threshold)).round(4).to_frame("value")
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
    st.markdown("#### 5 most important — what the model leans on")
    st.table(importance.head(5).round(3))
with weakest:
    st.markdown("#### 5 least important — candidates to untick and retrain without")
    st.table(importance.tail(5).round(3))
