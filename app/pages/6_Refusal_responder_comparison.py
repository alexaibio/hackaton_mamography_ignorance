import pandas as pd
import streamlit as st

from mammography.io import read_table
from mammography.model.train import load_run
from mammography.profiles import (
    feature_groups,
    most_important,
    normalised_groups,
    person_level,
    radar_figure,
)

PROFILE_FEATURES = 8
REFUSAL_LABEL = "refusal group"
RESPONDER_LABEL = "responder group"
COLOURS = {RESPONDER_LABEL: "seagreen", REFUSAL_LABEL: "crimson"}

st.set_page_config(page_title="Refusal/responder comparison", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def people(target: str) -> pd.DataFrame:
    return person_level(load_invitations(), target)


st.title("Refusal/responder comparison")

if "run" not in st.session_state:
    saved = load_run()
    if saved is not None:
        st.session_state["run"] = saved

run = st.session_state.get("run")
if run is None:
    st.info("No trained model in this session. Train one on the **Risk model** page first.")
    st.stop()

top_features = most_important(run, PROFILE_FEATURES)
groups = feature_groups(top_features)
everyone = people(run["target"])

refusers = everyone[everyone["response_rate"] == 0]
responders = everyone[everyone["response_rate"] > 0]

if refusers.empty or responders.empty:
    st.warning("One of the two groups is empty at this setting.")
    st.stop()

shape = normalised_groups(
    {REFUSAL_LABEL: refusers, RESPONDER_LABEL: responders}, everyone, groups
)

if shape.empty:
    st.info("No numeric or boolean features among the top ones to plot.")
    st.stop()

chart_column, legend_column = st.columns([3, 2])

with chart_column:
    st.pyplot(radar_figure(shape, COLOURS))

with legend_column:
    st.caption(
        f"Red is the {len(refusers):,} women who have never been screened after any "
        f"invitation, green the {len(responders):,} who have been screened after at "
        "least one. Every patient is in exactly one of the two groups. "
        f"The {len(top_features)} axes are the most important features in the trained "
        "model. Counts are divided by the whole population's 90th percentile so the axes "
        "share a scale; yes/no features are plotted as the share who are true."
    )
    st.caption(
        "Where green reaches outside red, that is the behaviour an intervention is "
        "trying to produce. Read these as markers of the habit, not as its causes."
    )
