import pandas as pd
import streamlit as st
from catboost import Pool

from mammography.config import get_config
from mammography.io import read_table
from mammography.llm.messages import draft_reminder_sequence
from mammography.model.train import MISSING, feature_columns, load_run

# history and demography cannot be changed by an intervention
FIXED_FEATURES = {
    "age_band",
    "vzp_territorial_area",
    "insurance_duration_years",
    "prior_invitation_count",
    "invitation_sequence",
    "months_last_invitation",
    "first_invitation",
    "prior_screen_count_5y",
    "months_since_last_screen",
    "never_screened_before_flag",
}

PERSON_LIST_LIMIT = 300

st.set_page_config(page_title="Refusal analysis", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def person_summary(target: str) -> pd.DataFrame:
    invitations = load_invitations().sort_values(["person_id", "invitation_date"])
    return invitations.groupby("person_id").agg(
        invitations=("invitation_id", "size"),
        responded=(target, "any"),
        first_invitation=("invitation_date", "min"),
        last_invitation=("invitation_date", "max"),
        age_band=("age_band", "last"),
        area=("vzp_territorial_area", "last"),
    )


@st.cache_data
def category_options(column: str) -> list[str]:
    values = load_invitations()[column].dropna().astype(str).unique()
    return sorted(values) + [MISSING]


def score(model, row: pd.DataFrame, categorical: list[str]) -> float:
    return float(model.predict_proba(Pool(row, cat_features=categorical))[0, 1])


st.title("Refusal analysis and interventions")

if "run" not in st.session_state:
    saved = load_run()
    if saved is not None:
        st.session_state["run"] = saved

run = st.session_state.get("run")
if run is None:
    st.info("No trained model in this session. Train one on the **Risk model** page first.")
    st.stop()

config = get_config()
_, _, boolean_features = feature_columns(config)
categorical = set(run["categorical"])
booleans = set(boolean_features) & set(run["features"])

invitations = load_invitations()
summary = person_summary(run["target"])
refusers = summary[~summary["responded"]]

st.caption(
    f"{len(refusers):,} of {len(summary):,} patients never had a screening within the "
    f"horizon of any invitation (`{run['target']}`)."
)

st.header("Select a patient")

band_column, area_column, sort_column = st.columns(3)
bands = band_column.multiselect("Age band", sorted(refusers["age_band"].unique()))
areas = area_column.multiselect("VZP area", sorted(refusers["area"].unique()))
most_invited = sort_column.checkbox("Most invited first", value=True)

filtered = refusers
if bands:
    filtered = filtered[filtered["age_band"].isin(bands)]
if areas:
    filtered = filtered[filtered["area"].isin(areas)]
if most_invited:
    filtered = filtered.sort_values("invitations", ascending=False)

if filtered.empty:
    st.warning("No patients match these filters.")
    st.stop()

st.caption(
    f"{len(filtered):,} patients match — the {PERSON_LIST_LIMIT} first are listed."
)
person_id = st.selectbox("Patient", filtered.index[:PERSON_LIST_LIMIT].tolist())

person = summary.loc[person_id]
history = invitations[invitations["person_id"] == person_id].sort_values("invitation_date")

st.header("Aggregated history")

first, second, third, fourth = st.columns(4)
first.metric("Invitations", int(person["invitations"]))
second.metric("Age band", person["age_band"])
third.metric(
    "Invited between",
    f"{person['first_invitation']:%b %Y} – {person['last_invitation']:%b %Y}",
)
fourth.metric("Screened in 5y before", int(history["prior_screen_count_5y"].iloc[-1]))

latest = history.iloc[-1]

context = pd.DataFrame(
    [
        ("VZP area", latest["vzp_territorial_area"]),
        ("Insured with VZP, years", latest["insurance_duration_years"]),
        ("Registered with a GP", latest["gp_registered"]),
        ("Registered with a gynaecologist", latest["gyn_registered_flag"]),
        ("Never screened before", latest["never_screened_before_flag"]),
        ("Healthcare contacts, 12m", latest["healthcare_contact_12m_cnt"]),
        ("GP prevention visits, 24m", latest["gp_prevention_24m_cnt"]),
        ("Gyn prevention visits, 24m", latest["gyn_prevention_24m_cnt"]),
        ("Other screening, 36m", latest["other_screening_36m_cnt"]),
        ("Dental prevention, 24m", latest["dental_prevention_24m_cnt"]),
    ],
    columns=["Attribute", "At the last invitation"],
).set_index("Attribute")

context_column, invitations_column = st.columns([2, 3])
context_column.table(context)
invitations_column.caption("Every invitation she received")
invitations_column.dataframe(
    history[
        [
            "invitation_date",
            "age_band",
            "invitation_sequence",
            "screening_3m_flag",
            "screening_6m_flag",
            "screening_12m_flag",
        ]
    ],
    width="stretch",
    hide_index=True,
)

baseline_row = latest[run["features"]].to_frame().T
for column in run["features"]:
    baseline_row[column] = (
        baseline_row[column].astype(str)
        if column in categorical
        else pd.to_numeric(baseline_row[column])
    )

baseline_risk = 1 - score(run["model"], baseline_row, run["categorical"])

st.header("What would change her mind?")
st.caption(
    "Only features an intervention could realistically move are editable. Age, history "
    "and insurance tenure are shown above and held fixed."
)

editable = [column for column in run["features"] if column not in FIXED_FEATURES]

if not editable:
    st.warning("None of the trained features are actionable.")
    st.stop()

with st.form("what_if_form"):
    changed: dict = {}
    widgets = st.columns(3)
    for position, column in enumerate(editable):
        widget = widgets[position % 3]
        current = latest[column]
        if column in categorical:
            options = category_options(column)
            index = options.index(str(current)) if str(current) in options else 0
            changed[column] = widget.selectbox(column, options, index=index)
        elif column in booleans:
            changed[column] = int(widget.checkbox(column, value=bool(current)))
        else:
            changed[column] = widget.number_input(
                column, min_value=0, value=int(current) if pd.notna(current) else 0, step=1
            )
    recalculated = st.form_submit_button("Recalculate the risk", type="primary")

if recalculated:
    st.session_state["what_if"] = changed

what_if = st.session_state.get("what_if")

risk_left, risk_middle, risk_right = st.columns(3)
risk_left.metric("Risk as she is now", f"{baseline_risk:.1%}")

if what_if is not None:
    modified_row = baseline_row.copy()
    for column, value in what_if.items():
        modified_row[column] = str(value) if column in categorical else value
    modified_risk = 1 - score(run["model"], modified_row, run["categorical"])
    risk_middle.metric(
        "Risk after the change",
        f"{modified_risk:.1%}",
        delta=f"{(modified_risk - baseline_risk) * 100:+.1f} pp",
        delta_color="inverse",
    )
    risk_right.metric("Change", f"{(baseline_risk - modified_risk) * 100:+.1f} pp avoided")

st.header("Reminder sequence")

channel = st.selectbox("Channel", ["letter", "sms", "email"])

if st.button("Create three consecutive reminders", type="primary"):
    patient = {
        "age_band": str(latest["age_band"]),
        "region": str(latest["vzp_territorial_area"]),
        "notification_channel": channel,
        "invitation_count": int(person["invitations"]),
        "gp_registered": bool(latest["gp_registered"]),
        "gyn_registered_flag": bool(latest["gyn_registered_flag"]),
        "never_screened_before_flag": bool(latest["never_screened_before_flag"]),
        "prior_screen_count_5y": int(latest["prior_screen_count_5y"]),
        "healthcare_contact_12m_cnt": int(latest["healthcare_contact_12m_cnt"]),
    }
    with st.spinner("Drafting the sequence..."):
        st.session_state["reminders"] = draft_reminder_sequence(patient, channel, count=3)

reminders = st.session_state.get("reminders")
if reminders:
    for position, text in enumerate(reminders, start=1):
        st.subheader(f"Reminder {position}")
        st.text_area(
            f"reminder_{position}", text, height=200, label_visibility="collapsed"
        )
    st.caption(
        "`[STATISTIKA]` and other bracketed placeholders are deliberate — the model is "
        "told never to invent a risk figure, date or phone number."
    )
