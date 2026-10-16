import pandas as pd
import streamlit as st

from mammography.io import read_table
from mammography.model.train import load_run
from mammography.profiles import category_gaps, contrast, feature_groups, person_level

st.set_page_config(page_title="What makes them different", layout="wide")


@st.cache_data
def load_invitations() -> pd.DataFrame:
    return read_table("interim", "invitations", parse_dates=["invitation_date"])


@st.cache_data
def people(target: str) -> pd.DataFrame:
    return person_level(load_invitations(), target)


st.title("What makes them different")

if "run" not in st.session_state:
    saved = load_run()
    if saved is not None:
        st.session_state["run"] = saved

run = st.session_state.get("run")
if run is None:
    st.info("No trained model in this session. Train one on the **Risk model** page first.")
    st.stop()

target = run["target"]
groups = feature_groups(run["features"])

st.caption(
    "The women who never come, set against the women who reliably do — across all "
    f"{len(run['features'])} features the model was trained on (`{target}`)."
)

everyone = people(target)

rate = st.slider(
    "Counts as reliable from this response rate upwards",
    0.3,
    1.0,
    0.5,
    0.05,
    help=(
        "invitations arrive yearly but screening is recommended every two years, so a "
        "woman following the guideline perfectly responds to about half of hers"
    ),
)

refusers = everyone[everyone["response_rate"] == 0]
reliable = everyone[everyone["response_rate"] >= rate]

if refusers.empty or reliable.empty:
    st.warning("One of the two groups is empty at this setting.")
    st.stop()

first, second, third = st.columns(3)
first.metric("Never respond", f"{len(refusers):,}")
second.metric(f"Respond to {rate:.0%} or more", f"{len(reliable):,}")
third.metric(
    "Everyone else",
    f"{len(everyone) - len(refusers) - len(reliable):,}",
    help="between the two definitions, excluded from this comparison",
)

st.header("Counts and yes/no attributes")
st.caption(
    "`this group` is the never-responders, `the others` the reliable ones. `effect` is "
    "the difference scaled by the spread across both groups, so rows are comparable; "
    "the table is sorted by it. Read these as markers, not causes."
)

differences = contrast(refusers, reliable, groups)
left, right = st.columns([3, 2])
left.table(differences)
right.bar_chart(differences["effect"])

st.header("Categories")
st.caption("Shares in %, with the gap in percentage points. Largest gaps first.")

for column in groups["categorical"]:
    st.subheader(column)
    st.table(category_gaps(refusers, reliable, column))
