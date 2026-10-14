import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.fhir.client import search_params
from mammography.io import read_table

FILTERED_RESOURCES = ("Patient", "Communication", "Appointment")
CHANNELS = ["has_phone", "has_mobile", "has_email", "has_registered_gp"]

st.set_page_config(page_title="Screening outreach", layout="wide")


@st.cache_data
def load_patients() -> pd.DataFrame:
    patients = read_table("interim", "patients", parse_dates=["birth_date", "deceased_date"])
    today = pd.Timestamp("today").normalize()
    patients["age"] = ((today - patients["birth_date"]).dt.days / 365.25).astype(int)
    return patients


st.title("Mammography screening outreach")
st.caption("Synthetic demo data — not real screening outcomes.")

config = get_config()
fetch_config = config.get("fetch", {})
patient_filter = fetch_config.get("patient", {})
history_years = fetch_config.get("history_years")

st.header("What data do we pull?")

criteria = pd.DataFrame(
    [
        ("Gender", patient_filter.get("gender", "all")),
        ("Minimum age at download", patient_filter.get("min_age", "no limit")),
        ("History depth", f"last {history_years} years" if history_years else "all history"),
    ],
    columns=["Criterion", "Value"],
)
st.table(criteria.set_index("Criterion"))

resolved = pd.DataFrame(
    [
        (resource_type, ", ".join(f"{k}={v}" for k, v in search_params(resource_type).items()))
        for resource_type in FILTERED_RESOURCES
    ],
    columns=["Resource", "FHIR search filter"],
)
st.caption("Server-side filters actually applied when the data was downloaded:")
st.table(resolved.set_index("Resource"))

st.header("What did we extract?")

patients = load_patients()
min_age = patient_filter.get("min_age", 0)
gender = patient_filter.get("gender")

matching = patients["age"] >= min_age
if gender:
    matching &= patients["gender"] == gender
matching &= patients["deceased_date"].isna()

reachable = patients[["has_phone", "has_mobile", "has_email"]].any(axis=1)

first, second, third, fourth = st.columns(4)
first.metric("Patients extracted", len(patients))
second.metric("Alive and matching the filter", int(matching.sum()))
third.metric("Deceased", int(patients["deceased_date"].notna().sum()))
fourth.metric("No contact channel at all", int((~reachable).sum()))

st.subheader("Age")
st.bar_chart(patients["age"].value_counts().sort_index())

left, right = st.columns(2)

with left:
    st.subheader("Contactability")
    st.table(patients[CHANNELS].mean().round(3).rename("share").to_frame())

    st.subheader("Completeness")
    st.dataframe(
        patients.isna().mean().sort_values(ascending=False).round(3).rename("null rate").to_frame(),
        width="stretch",
    )

with right:
    for column in ["state", "district", "marital_status", "preferred_language"]:
        st.subheader(column)
        st.table(patients[column].value_counts(dropna=False).head(8).rename("patients").to_frame())
