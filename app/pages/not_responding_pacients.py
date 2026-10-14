import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table
from mammography.model.predict import score_cohort

st.set_page_config(page_title="Not responding patients", layout="wide")


@st.cache_data
def load_scored() -> pd.DataFrame:
    config = get_config()
    invitations = read_table("processed", config["modelling_table"])
    return score_cohort(invitations)


st.title("Patients unlikely to respond")
st.caption("Synthetic demo data — identifiers are pseudonymous.")

scored = load_scored()

threshold = st.slider("Minimum risk", 0.0, 1.0, 0.5, 0.05)
top_n = st.number_input("Rows to list", min_value=10, max_value=5000, value=100, step=10)

filters = scored["risk"] >= threshold

for column in ("region", "notification_channel"):
    if column in scored.columns:
        values = sorted(scored[column].dropna().unique())
        chosen = st.multiselect(column, values, default=values)
        filters &= scored[column].isin(chosen)

worklist = scored.loc[filters].sort_values("risk", ascending=False).head(int(top_n))

st.write(f"{int(filters.sum())} patients above threshold, showing {len(worklist)}")
st.dataframe(worklist, width="stretch")

st.download_button(
    "Download as CSV",
    worklist.to_csv(index=False).encode(),
    file_name="not_responding_patients.csv",
    mime="text/csv",
)
