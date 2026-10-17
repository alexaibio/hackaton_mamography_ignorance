import pandas as pd
import streamlit as st
from catboost import Pool

from mammography.config import get_config
from mammography.interventions import TEXTS_FILE, proposed_interventions
from mammography.io import read_table
from mammography.model.train import MISSING, feature_columns, load_run, saved_targets

TOP_DRIVERS = 5

HORIZON_LABELS = {
    "screening_3m_flag": "within 3 months",
    "screening_6m_flag": "within 6 months",
    "screening_12m_flag": "within 12 months",
}
UNLIKELY_AT_ANY_HORIZON = 0.20
LATE_RESPONDER_GAP = 0.10

st.set_page_config(page_title="Patient risk", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def intervention_texts() -> dict[str, str]:
    return proposed_interventions()


@st.cache_data
def category_options(column: str) -> list[str]:
    values = load_invitations()[column].dropna().astype(str).unique()
    return sorted(values) + [MISSING]


@st.cache_data
def numeric_defaults() -> dict[str, int]:
    invitations = load_invitations()
    _, numeric, _ = feature_columns(get_config())
    medians = invitations[numeric].median()
    return {column: int(value) if pd.notna(value) else 0 for column, value in medians.items()}


st.title("Patient risk scoring")

run = st.session_state.get("run")
if run is None:
    st.info("No trained model in this session. Train one on the **Risk model** page first.")
    st.stop()

st.caption(
    f"Model trained on label `{run['target']}` with {len(run['features'])} features. "
    "Enter a patient below to score her."
)

config = get_config()
_, _, boolean_features = feature_columns(config)
categorical = set(run["categorical"])
booleans = set(boolean_features) & set(run["features"])
defaults = numeric_defaults()

st.header("New patient")

with st.form("patient_form"):
    entered: dict = {}
    widgets = st.columns(3)

    for position, column in enumerate(run["features"]):
        widget = widgets[position % 3]
        if column in categorical:
            entered[column] = widget.selectbox(column, category_options(column))
        elif column in booleans:
            entered[column] = int(widget.checkbox(column, value=True))
        else:
            entered[column] = widget.number_input(
                column, min_value=0, value=defaults.get(column, 0), step=1
            )

    scored = st.form_submit_button("Calculate the probability", type="primary")

if scored:
    st.session_state["patient_input"] = entered

patient = st.session_state.get("patient_input")
if patient is None:
    st.stop()

row = pd.DataFrame([patient])[run["features"]]
for column in run["features"]:
    row[column] = row[column].astype(str) if column in categorical else row[column]

pool = Pool(row, cat_features=run["categorical"])
response_probability = float(run["model"].predict_proba(pool)[0, 1])
risk = 1 - response_probability

st.header("Prediction")

left, right = st.columns(2)
left.metric("Probability she responds", f"{response_probability:.1%}")
right.metric("Risk of ignoring", f"{risk:.1%}")

st.progress(risk, text=f"risk of ignoring: {risk:.1%}")

st.header("Response speed")

horizon_runs = {}
for horizon in HORIZON_LABELS:
    other = run if horizon == run["target"] else load_run(horizon)
    if other is not None and set(other["features"]) <= set(patient):
        horizon_runs[horizon] = other

if len(horizon_runs) < len(HORIZON_LABELS):
    st.info(
        "Needs a model for all three horizons. On the **Risk model** page tick "
        "*Also train the other two horizons* and retrain. "
        f"Currently saved: {', '.join(f'`{name}`' for name in saved_targets()) or 'none'}."
    )
else:
    speeds = {}
    for horizon, other in horizon_runs.items():
        horizon_row = pd.DataFrame([patient])[other["features"]]
        for column in other["categorical"]:
            horizon_row[column] = horizon_row[column].astype(str)
        speeds[horizon] = float(
            other["model"].predict_proba(
                Pool(horizon_row, cat_features=other["categorical"])
            )[0, 1]
        )

    three, six, twelve = (
        speeds["screening_3m_flag"],
        speeds["screening_6m_flag"],
        speeds["screening_12m_flag"],
    )

    if twelve < UNLIKELY_AT_ANY_HORIZON:
        verdict = "Unlikely to attend at any horizon"
        advice = (
            "a reminder on its own will probably not move her — the GP or gynaecologist "
            "route is the better lever"
        )
    elif twelve - three >= LATE_RESPONDER_GAP:
        verdict = "Late responder"
        advice = (
            "she does attend, but not promptly — timing is the lever, so a reminder a few "
            "weeks after the invitation is worth more than a stronger first letter"
        )
    else:
        verdict = "Prompt responder"
        advice = "if she attends at all she attends early; no extra contact needed"

    speed_columns = st.columns(3)
    for column, (horizon, probability) in zip(speed_columns, speeds.items()):
        column.metric(HORIZON_LABELS[horizon], f"{probability:.1%}")

    st.markdown(f"#### {verdict}")
    st.caption(f"Probability of responding by each horizon. {advice}.")
    st.bar_chart(
        pd.Series(
            {HORIZON_LABELS[horizon]: value for horizon, value in speeds.items()},
            name="probability of responding",
        )
    )

    if not three <= six <= twelve:
        st.caption(
            "The three horizons are modelled independently, so their probabilities are "
            "not forced to increase. The ordering is broken here, which usually means "
            "the models disagree about a patient in a thin region of the data."
        )

st.header("What drives this prediction")

shap_values = run["model"].get_feature_importance(type="ShapValues", data=pool)
contributions = pd.Series(shap_values[0, :-1], index=run["features"])
ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

top = ranked.head(TOP_DRIVERS)
texts = intervention_texts()

drivers = pd.DataFrame(
    {
        "value": [patient[name] for name in top.index],
        "effect on responding": top.round(4),
        "direction": [
            "raises the chance she responds" if value > 0 else "raises the risk of ignoring"
            for value in top
        ],
        "proposed intervention": [texts.get(name, "") for name in top.index],
    }
)
st.table(drivers)
st.caption(
    "SHAP values for this single patient — log-odds contribution of each feature to the "
    "probability of responding. Positive pushes towards responding."
)

if not texts:
    st.warning(
        f"No intervention texts loaded. Expected `{TEXTS_FILE}` next to the "
        "`mammography` package or in the data folder, with the feature name in the "
        "first column and the text to show in the second."
    )
else:
    unmapped = [name for name in top.index if name not in texts]
    if unmapped:
        st.caption(f"No intervention text for: `{'`, `'.join(unmapped)}`")

