import html

import pandas as pd
import streamlit as st
from catboost import Pool

from mammography.config import get_config
from mammography.interventions import TEXTS_FILE, proposed_interventions
from mammography.io import read_table
from mammography.llm.messages import (
    clinician_messages,
    draft_clinician_email,
    draft_reminder_sequence,
    reminder_messages,
    render_messages,
)
from mammography.model.train import MISSING, feature_columns, load_run

CANDIDATE_LIMIT = 2000
MAX_STEPS = 3

# how each proposed change reads in a sentence
ACTION_PHRASES = {
    "gyn_registered_flag": "register her with a gynaecologist",
    "gp_registered": "register her with a general practitioner",
    "gyn_prevention_24m_cnt": "bring her to one preventive visit at the gynaecologist",
    "gp_prevention_24m_cnt": "bring her to one preventive visit at the GP",
    "other_screening_36m_cnt": "invite her to one other screening programme",
    "dental_prevention_24m_cnt": "get her to one dental preventive check-up",
    "healthcare_contact_12m_cnt": "get her one more contact with any doctor",
}

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
def intervention_texts() -> dict[str, str]:
    return proposed_interventions()


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

st.header("Choose the patient with high risk of not responding")

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
    height=215,
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

st.header(f"Selected patient {person_id}")

person = summary.loc[person_id]
history = invitations[invitations["person_id"] == person_id].sort_values("invitation_date")
latest = history.iloc[-1]

first, second, third, fourth = st.columns([1, 1, 2.4, 1.4])
first.metric("Invitations", int(person["invitations"]))
second.metric("Age band", person["age_band"])
third.metric(
    "Invited between",
    f"{person['first_invitation']:%m/%Y} – {person['last_invitation']:%m/%Y}",
)
screened_before = int(latest["prior_screen_count_5y"])
with fourth:
    st.caption("Screened in 5y before")
    st.markdown(f"### :red[{screened_before}]")
    if screened_before == 0:
        st.caption(":red[never screened — the clearest signal she is not attending]")

registered_with_gp = (
    f"registered with a general practitioner (**{latest['gp_reg_duration_band'] or 'n/a'}** years)"
    if latest["gp_registered"]
    else "**not registered** with any general practitioner"
)
registered_with_gyn = (
    f"registered with a gynaecologist (**{latest['gyn_reg_duration_band'] or 'n/a'}** years)"
    if latest["gyn_registered_flag"]
    else "**not registered** with any gynaecologist"
)
screening_history = (
    "has **never** had a mammography screening"
    if latest["never_screened_before_flag"]
    else f"was screened **{int(latest['prior_screen_count_5y'])}** times in the last five years"
)

description = (
    f"She lives in the **{latest['vzp_territorial_area']}** area and has been insured with "
    f"VZP for **{int(latest['insurance_duration_years'])}** years. She is "
    f"{registered_with_gp} and {registered_with_gyn}, and {screening_history}. In the twelve "
    f"months before her last invitation she had **{int(latest['healthcare_contact_12m_cnt'])}** "
    f"contacts with a doctor. Over twenty-four months she had "
    f"**{int(latest['gp_prevention_24m_cnt'])}** preventive check-ups at the general "
    f"practitioner, **{int(latest['gyn_prevention_24m_cnt'])}** at the gynaecologist and "
    f"**{int(latest['dental_prevention_24m_cnt'])}** at the dentist, and over thirty-six "
    f"months she took part in **{int(latest['other_screening_36m_cnt'])}** other screening "
    "programmes."
)

st.caption("At her last invitation")
st.write(description)

st.caption("Every invitation she received")
st.dataframe(
    history[["invitation_date", "age_band", "invitation_sequence"]],
    width="stretch",
    hide_index=True,
)

st.header("Proposed interventions to motivate to come to screening")

actionable = [column for column in run["features"] if column in IMPROVEMENTS]

st.caption(
    f"Only {len(actionable)} of the model's {len(run['features'])} features can be moved "
    "by an intervention. The rest are demography or history and are held fixed."
)

baseline_row = feature_frame(
    latest.to_frame().T, run["features"], categorical, booleans
)
baseline_risk = float(refusal_risk(run["model"], baseline_row, run["categorical"])[0])

steps, final_risk = propose_changes(
    run["model"], baseline_row, run["categorical"], run["features"]
)

texts = intervention_texts()

if not steps:
    st.warning(
        "No single actionable feature reduces this patient's risk. Everything the model "
        "relies on for her is history or demography, which an intervention cannot change."
    )
    st.metric("Risk of refusal, unchanged", f"{baseline_risk:.1%}")
else:
    st.markdown("#### :red[What we propose for her (with all proposals)]")
    st.markdown(
        "\n".join(
            f"{position}. :red[{ACTION_PHRASES.get(step['feature'], step['feature'])}]"
            for position, step in enumerate(steps, start=1)
        )
    )
    st.caption(
        f"{len(steps)} change"
        + ("s" if len(steps) > 1 else "")
        + ", in the order that reduces risk fastest. Each row shows the risk once that "
        "change and every change above it has been made."
    )
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
            f"No intervention texts loaded — add `{TEXTS_FILE}` next to the "
            "`mammography` package with the feature name in the first column."
        )

    current_column, reduction_column, after_column = st.columns(3)
    current_column.metric("Risk of refusal as she is now", f"{baseline_risk:.1%}")
    reduction_column.metric(
        "Reduction from the changes above", f"{(baseline_risk - final_risk) * 100:.1f} pp"
    )
    after_column.metric(
        f"Risk after all {len(steps)} changes above",
        f"{final_risk:.1%}",
        delta=f"{(final_risk - baseline_risk) * 100:+.1f} pp",
        delta_color="inverse",
    )
    st.caption(
        "The model's estimate of her risk if her record looked that way instead. It is an "
        "association the model learned, not a guarantee that making these changes causes "
        "the drop."
    )

CLINICIANS = (
    ("gynaecologist", "gyn_registered_flag", "gyn_email"),
    ("general practitioner", "gp_registered", "gp_email"),
)

patient = latest.to_dict()
patient["invitation_count"] = int(person["invitations"])
if steps:
    patient["suggested_actions"] = [texts.get(step["feature"], step["feature"]) for step in steps]

st.header("Reminder to patient")

REMINDER_PURPOSE = (
    "sent first — make attending feel easy",
    "sent a few weeks later — remove her specific obstacle",
    "sent last — offer to do the work for her",
)

reminders = st.session_state.get("reminders")
if not reminders:
    st.info("Press **Generate all texts** at the bottom of the page.")
else:
    for position, text in enumerate(reminders, start=1):
        purpose = (
            REMINDER_PURPOSE[position - 1] if position <= len(REMINDER_PURPOSE) else ""
        )
        with st.container(border=True):
            st.markdown(f"### Reminder {position}")
            st.caption(purpose)
            st.markdown(
                "<div style='font-size:1.35rem; line-height:1.65; white-space:pre-wrap'>"
                f"{html.escape(text)}</div>",
                unsafe_allow_html=True,
            )
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

doctor_columns = st.columns(2)

for (clinician, flag, state_key), column in zip(CLINICIANS, doctor_columns):
    with column:
        st.subheader(f"To her {clinician}")
        if not bool(latest[flag]):
            st.warning(
                f"She has no registered {clinician}, so there is nobody to write to. "
                "Helping her register is itself the intervention."
            )
            continue

        email = st.session_state.get(state_key)
        if not email:
            st.info("Press **Generate all texts** at the bottom of the page.")
            continue

        st.text_area(state_key, email, height=320, label_visibility="collapsed")

st.divider()
st.header("Generate")

channel = st.selectbox("Channel for the patient messages", ["letter", "sms", "email"])
patient["notification_channel"] = channel

recipients = [
    (clinician, state_key)
    for clinician, flag, state_key in CLINICIANS
    if bool(latest[flag])
]

with st.expander("What is sent to the model"):
    st.caption("The patient reminders")
    st.code(render_messages(reminder_messages(patient, channel, count=3)), language=None)
    for clinician, _ in recipients:
        st.caption(f"The email to her {clinician}")
        st.code(render_messages(clinician_messages(patient, clinician)), language=None)

if st.button("Generate all texts", type="primary"):
    with st.spinner("Drafting the three reminders..."):
        st.session_state["reminders"] = draft_reminder_sequence(patient, channel, count=3)
    for clinician, state_key in recipients:
        with st.spinner(f"Drafting the email to her {clinician}..."):
            st.session_state[state_key] = draft_clinician_email(patient, clinician)
    st.rerun()
