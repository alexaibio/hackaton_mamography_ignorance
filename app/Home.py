import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mammography.io import read_table

CHANNELS = ["has_phone", "has_mobile", "has_email", "has_registered_gp"]
PLACEHOLDER = "Illustrative numbers — replaced once the invitation layer and labels exist."

st.set_page_config(page_title="Screening outreach", layout="wide")


@st.cache_data
def load_patients() -> pd.DataFrame:
    patients = read_table("interim", "patients", parse_dates=["birth_date", "deceased_date"])
    today = pd.Timestamp("today").normalize()
    patients["age"] = ((today - patients["birth_date"]).dt.days / 365.25).astype(int)
    return patients


def draw_pie(values: dict[str, int], title: str) -> None:
    figure, axes = plt.subplots(figsize=(4, 4))
    axes.pie(values.values(), labels=values.keys(), autopct="%1.0f%%", startangle=90)
    axes.set_title(title)
    axes.axis("equal")
    st.pyplot(figure)


st.title("Mammography screening outreach")
st.caption("Synthetic demo data — not real screening outcomes.")

patients = load_patients()
reachable = patients[["has_phone", "has_mobile", "has_email"]].any(axis=1)

first, second, third, fourth = st.columns(4)
first.metric("Patients extracted", len(patients))
third.metric("Deceased", int(patients["deceased_date"].notna().sum()))
second.metric("Alive", int(patients["deceased_date"].isna().sum()))
fourth.metric("No contact channel at all", int((~reachable).sum()))

st.header("Invitation response")
st.info(PLACEHOLDER)

left, right = st.columns(2)
with left:
    draw_pie(
        {"Attended": 38, "Did not respond": 52, "No outcome yet": 10},
        "Invitation outcome",
    )
with right:
    draw_pie(
        {"Letter": 55, "SMS": 25, "Email": 12, "Phone": 8},
        "Non-responders by last channel used",
    )

st.header("What did we extract?")

st.subheader("Age")
st.bar_chart(patients["age"].value_counts().sort_index())

stats_left, stats_right = st.columns(2)

with stats_left:
    st.subheader("Contactability")
    st.table(patients[CHANNELS].mean().round(3).rename("share").to_frame())

    st.subheader("Completeness")
    st.dataframe(
        patients.isna().mean().sort_values(ascending=False).round(3).rename("null rate").to_frame(),
        width="stretch",
    )

with stats_right:
    for column in ["state", "district", "marital_status", "preferred_language"]:
        st.subheader(column)
        st.table(patients[column].value_counts(dropna=False).head(8).rename("patients").to_frame())
