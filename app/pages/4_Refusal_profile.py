import pandas as pd
import streamlit as st

from mammography.io import read_table
from mammography.model.train import load_run
from mammography.profiles import feature_groups, most_important, person_level, profile

PROFILE_FEATURES = 8

st.set_page_config(page_title="Refusal profile", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def people(target: str) -> pd.DataFrame:
    return person_level(load_invitations(), target)


st.title("Refusal patient profile")

if "run" not in st.session_state:
    saved = load_run()
    if saved is not None:
        st.session_state["run"] = saved

run = st.session_state.get("run")
if run is None:
    st.info("No trained model in this session. Train one on the **Risk model** page first.")
    st.stop()

target = run["target"]
top_features = most_important(run, PROFILE_FEATURES)
groups = feature_groups(top_features)

st.caption(
    f"Women who were never screened within the horizon of any invitation (`{target}`). "
    f"The {len(top_features)} features shown are the most important ones in the trained "
    "model, out of the {total} it uses.".format(total=len(run["features"]))
)

everyone = people(target)
refusers = everyone[everyone["response_rate"] == 0]

if refusers.empty:
    st.warning("No patients match this definition.")
    st.stop()

st.header("The typical refusing patient")
st.caption(
    "Median for counts, share for yes/no attributes, most common value for categories. "
    "A profile of medians describes a type, not a person — the portrait below is a real "
    "patient."
)
st.table(profile(refusers, groups))
