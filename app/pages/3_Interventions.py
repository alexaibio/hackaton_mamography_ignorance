import pandas as pd
import streamlit as st
from catboost import Pool

from mammography.config import get_config, get_settings
from mammography.io import read_table
from mammography.llm.messages import (
    describe_patient,
    draft_clinician_email,
    draft_reminder_sequence,
)
from mammography.model.train import MISSING, feature_columns, load_run

INTERVENTIONS_FILE = "proposed_correction.csv"
CANDIDATE_LIMIT = 2000
MAX_STEPS = 3

# what an intervention could realistically move, and the value it would move it to
IMPROVEMENTS = {
    "gyn_registered_flag": lambda current: 1,
    "gp_registered": lambda current: 1,
    "gyn_prevention_24m_cnt": lambda current: int(current) + 1,
    "gp_prevention_24m_cnt": lambda current: int(current) + 1,
    "other_screening_36m_cnt": lambda current: int(current) + 1,
    "dental_prevention_24m_cnt": lambda current: int(current) + 1,
    "healthcare_contact_12m_cnt": lambda current: int(current) + 1,
}

st.set_page_config(page_title="Interventions", layout="wide")


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
def proposed_interventions() -> dict[str, str]:
    path = get_settings().data_folder / INTERVENTIONS_FILE
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    feature_column, text_column = table.columns[:2]
    return dict(zip(table[feature_column].astype(str), table[text_column].astype(str)))


def feature_frame(rows: pd.DataFrame, features: list[str], categorical: set, booleans: set):
    frame = rows[features].copy()
    for column in features:
        if column in categorical:
            frame[column] = frame[column].fillna(MISSING).astype(str)
        elif column in booleans:
            frame[column] = frame[column].astype("int8")
        else:
            frame[column] = pd.to_numeric(frame[column])
    return frame


def refusal_risk(model, frame: pd.DataFrame, categorical: list[str]):
    return 1 - model.predict_proba(Pool(frame, cat_features=categorical))[:, 1]


def propose_changes(model, row: pd.DataFrame, categorical: list[str], features: list[str]):
    current = row.copy()
    current_risk = float(refusal_risk(model, current, categorical)[0])
    steps = []

    for _ in range(MAX_STEPS):
        best = None
        for feature, improve in IMPROVEMENTS.items():
            if feature not in features or feature in {step["feature"] for step in steps}:
                continue
            present = current[feature].iloc[0]
            proposed = improve(present)
            if proposed == present:
                continue
            candidate = current.copy()
            candidate[feature] = proposed
            risk_after = float(refusal_risk(model, candidate, categorical)[0])
            if best is None or risk_after < best["risk_after"]:
                best = {
                    "feature": feature,
                    "from": present,
                    "to": proposed,
                    "risk_after": risk_after,
                }
        if best is None or best["risk_after"] >= current_risk - 1e-6:
            break
        current[best["feature"]] = best["to"]
        current_risk = best["risk_after"]
        steps.append(best)

    return steps, current_risk


st.title("Interventions")

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

band_column, area_column = st.columns(2)
bands = band_column.multiselect("Age band", sorted(refusers["age_band"].unique()))
areas = area_column.multiselect("VZP area", sorted(refusers["area"].unique()))

filtered = refusers
if bands:
    filtered = filtered[filtered["age_band"].isin(bands)]
if areas:
    filtered = filtered[filtered["area"].isin(areas)]

if filtered.empty:
    st.warning("No patients match these filters.")
    st.stop()

candidates = filtered.index[:CANDIDATE_LIMIT]
latest_rows = (
    invitations[invitations["person_id"].isin(candidates)]
    .sort_values(["person_id", "invitation_date"])
    .groupby("person_id")
    .tail(1)
    .set_index("person_id")
)

scored = pd.DataFrame(
    {
        "risk_of_refusal": refusal_risk(
            run["model"],
            feature_frame(latest_rows, run["features"], categorical, booleans),
            run["categorical"],
        )
    },
    index=latest_rows.index,
).join(filtered[["invitations", "age_band", "area", "last_invitation"]])

scored = scored.sort_values("risk_of_refusal", ascending=False).round({"risk_of_refusal": 4})

st.caption(
    f"{len(filtered):,} patients match the filters; the first {len(scored):,} are scored "
    "and ranked by risk of refusal. Click a row to select."
)

selection = st.dataframe(
    scored,
    width="stretch",
    height=360,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "risk_of_refusal": st.column_config.ProgressColumn(
            "risk of refusal", min_value=0.0, max_value=1.0, format="%.3f"
        )
    },
)

chosen_rows = selection.selection.rows
position = chosen_rows[0] if chosen_rows else 0
person_id = scored.index[position]

st.header(f"Patient {person_id}")

person = summary.loc[person_id]
history = invitations[invitations["person_id"] == person_id].sort_values("invitation_date")
latest = history.iloc[-1]

first, second, third, fourth = st.columns(4)
first.metric("Invitations", int(person["invitations"]))
second.metric("Age band", person["age_band"])
third.metric(
    "Invited between",
    f"{person['first_invitation']:%b %Y} – {person['last_invitation']:%b %Y}",
)
fourth.metric("Screened in 5y before", int(latest["prior_screen_count_5y"]))

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

st.header("Proposed intervention")

actionable = [column for column in run["features"] if column in IMPROVEMENTS]
held_fixed = [column for column in run["features"] if column not in IMPROVEMENTS]

st.caption(
    f"Only {len(actionable)} of the model's {len(run['features'])} features can be moved "
    "by an intervention. The rest are demography or history and are held fixed."
)

fixed_left, fixed_right = st.columns(2)
fixed_left.success("**Can be changed**\n\n" + "\n".join(f"- `{name}`" for name in actionable))
fixed_right.error(
    "**Held fixed** — age, region, insurance tenure and screening history\n\n"
    + "\n".join(f"- `{name}`" for name in held_fixed)
)

baseline_row = feature_frame(
    latest.to_frame().T, run["features"], categorical, booleans
)
baseline_risk = float(refusal_risk(run["model"], baseline_row, run["categorical"])[0])

steps, final_risk = propose_changes(
    run["model"], baseline_row, run["categorical"], run["features"]
)

current_column, arrow_column, after_column = st.columns(3)
current_column.metric("Current risk of refusal", f"{baseline_risk:.1%}")
after_column.metric(
    "Risk after the intervention",
    f"{final_risk:.1%}",
    delta=f"{(final_risk - baseline_risk) * 100:+.1f} pp",
    delta_color="inverse",
)
arrow_column.metric("Reduction", f"{(baseline_risk - final_risk) * 100:.1f} pp")

texts = proposed_interventions()

if not steps:
    st.warning(
        "No single actionable feature reduces this patient's risk. Everything the model "
        "relies on for her is history or demography, which an intervention cannot change."
    )
else:
    st.caption("Features to change, in the order that reduces risk fastest:")
    st.table(
        pd.DataFrame(
            [
                {
                    "feature": step["feature"],
                    "from": step["from"],
                    "to": step["to"],
                    "risk after this step": f"{step['risk_after']:.1%}",
                    "proposed intervention": texts.get(step["feature"], ""),
                }
                for step in steps
            ]
        ).set_index("feature")
    )
    if not texts:
        st.caption(
            f"No intervention texts loaded — add `{INTERVENTIONS_FILE}` to "
            f"`{get_settings().data_folder}` with the feature name in the first column."
        )

st.header("Reminder to patient")

channel = st.selectbox("Channel", ["letter", "sms", "email"])

patient = latest.to_dict()
patient["notification_channel"] = channel
patient["invitation_count"] = int(person["invitations"])
if steps:
    patient["suggested_actions"] = [texts.get(step["feature"], step["feature"]) for step in steps]

with st.expander("What is sent to the model"):
    st.json(describe_patient(patient))

if st.button("Create three consecutive reminders", type="primary"):
    with st.spinner("Drafting the sequence..."):
        st.session_state["reminders"] = draft_reminder_sequence(patient, channel, count=3)

reminders = st.session_state.get("reminders")
if reminders:
    for position, text in enumerate(reminders, start=1):
        st.subheader(f"Reminder {position}")
        st.text_area(f"reminder_{position}", text, height=200, label_visibility="collapsed")
    st.caption(
        "`[STATISTIKA]` and other bracketed placeholders are deliberate — the model is "
        "told never to invent a risk figure, date or phone number."
    )

st.header("Email to her doctors")
st.caption(
    "Asking the physicians she already sees to raise the screening at her next visit. "
    "The patient's name and insurance number are left as placeholders — the sending "
    "system fills those in, they are never passed to the model."
)

CLINICIANS = (
    ("gynaecologist", "gyn_registered_flag", "gyn_email"),
    ("general practitioner", "gp_registered", "gp_email"),
)

doctor_columns = st.columns(2)

for (clinician, flag, state_key), column in zip(CLINICIANS, doctor_columns):
    with column:
        st.subheader(f"To her {clinician}")
        registered = bool(latest[flag])

        if not registered:
            st.warning(
                f"She has no registered {clinician}, so there is nobody to write to. "
                "Helping her register is itself the intervention."
            )
            continue

        if st.button(f"Draft the email to her {clinician}", key=f"button_{state_key}"):
            with st.spinner("Drafting..."):
                st.session_state[state_key] = draft_clinician_email(patient, clinician)

        st.text_area(
            state_key,
            st.session_state.get(state_key, ""),
            height=300,
            label_visibility="collapsed",
        )
