import pandas as pd
import streamlit as st

from mammography.config import get_config
from mammography.io import read_table
from mammography.model.predict import score_cohort

st.set_page_config(page_title="Screening outreach", layout="wide")


@st.cache_data
def load_scored() -> pd.DataFrame:
    config = get_config()
    invitations = read_table("processed", config["modelling_table"])
    return score_cohort(invitations)


st.title("Mammography screening outreach")
st.caption("Synthetic demo data — the numbers below are not real screening outcomes.")

config = get_config()
target = config["target"]
scored = load_scored()

st.header("What does targeted outreach buy us?")

budget = st.slider(
    "How many patients can we contact?",
    min_value=0,
    max_value=len(scored),
    value=min(500, len(scored)),
    step=10,
)

ranked = scored.sort_values("risk", ascending=False)
ignore_rate = (scored[target] == 0).mean()
targeted = int((ranked.head(budget)[target] == 0).sum())
random_pick = int(round(budget * ignore_rate))

left, middle, right = st.columns(3)
left.metric("Non-attenders reached, model-targeted", targeted)
middle.metric("Non-attenders reached, contacted at random", random_pick)
right.metric("Additional non-attenders reached", targeted - random_pick)

st.header("Cumulative gain")

gains = pd.DataFrame(
    {
        "model": (ranked[target] == 0).cumsum().values,
        "random": (pd.Series(range(1, len(ranked) + 1)) * ignore_rate).values,
    }
)
st.line_chart(gains)

st.caption(
    f"Cohort of {len(scored)} invitations, baseline non-attendance {ignore_rate:.1%}."
)
