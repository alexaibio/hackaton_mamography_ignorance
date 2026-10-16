import pandas as pd
import streamlit as st
from catboost import Pool

from mammography.config import get_config, get_settings
from mammography.io import read_table
from mammography.model.train import MISSING, feature_columns

INTERVENTIONS_FILE = "proposed_correction.csv"
TOP_DRIVERS = 5

st.set_page_config(page_title="Patient risk", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def proposed_interventions() -> dict[str, str]:
    path = get_settings().data_folder / INTERVENTIONS_FILE
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    feature_column, text_column = table.columns[:2]
    return dict(zip(table[feature_column].astype(str), table[text_column].astype(str)))


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

test_risk = 1 - run["screening_probability"]
percentile = float((test_risk < risk).mean())

st.header("Prediction")

left, middle, right = st.columns(3)
left.metric("Probability she responds", f"{response_probability:.1%}")
middle.metric("Risk of ignoring", f"{risk:.1%}")
right.metric(
    "Riskier than",
    f"{percentile:.0%} of the test set",
    help="where this patient sits among the held-out invitations, ranked by risk",
)

st.progress(risk, text=f"risk of ignoring: {risk:.1%}")

st.header("What drives this prediction")

shap_values = run["model"].get_feature_importance(type="ShapValues", data=pool)
contributions = pd.Series(shap_values[0, :-1], index=run["features"])
ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

top = ranked.head(TOP_DRIVERS)
texts = proposed_interventions()

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
        f"No intervention texts loaded. Expected `{INTERVENTIONS_FILE}` in "
        f"`{get_settings().data_folder}`, with the feature name in the first column "
        "and the text to show in the second."
    )
else:
    unmapped = [name for name in top.index if name not in texts]
    if unmapped:
        st.caption(f"No intervention text for: `{'`, `'.join(unmapped)}`")

