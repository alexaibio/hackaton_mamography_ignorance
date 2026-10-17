import pandas as pd
import streamlit as st

from mammography.clusters import cluster_descriptions, cluster_figure, shap_clusters
from mammography.io import read_table
from mammography.model.train import load_run
from mammography.profiles import (
    feature_groups,
    most_important,
    normalised_profile,
    person_level,
    profile,
    radar_figure,
)

PROFILE_FEATURES = 8
GROUP_LABEL = "refusal group"

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

st.header("The typical refusing patient at a glance")

shape = normalised_profile(refusers, everyone, groups, GROUP_LABEL)

if shape.empty:
    st.info("No numeric or boolean features among the top ones to plot.")
else:
    chart_column, legend_column = st.columns([3, 2])

    with chart_column:
        st.pyplot(radar_figure(shape, {"all patients": "tab:gray", GROUP_LABEL: "crimson"}))

    with legend_column:
        st.caption(
            "Red is the women who never come, grey is every patient in the data. "
            "Counts are divided by the population's 90th percentile so the axes share a "
            "scale; yes/no features are plotted as the share who are true. Where red "
            "sits inside grey, the refusal group does less of that than the population."
        )
        st.dataframe(shape.round(3), width="stretch")

    st.caption("Categories cannot go on a radar, so the most common values are:")
    st.table(
        pd.DataFrame(
            [
                (
                    column,
                    refusers[column].value_counts(dropna=False).index[0],
                    f"{refusers[column].value_counts(normalize=True, dropna=False).iloc[0]:.0%}",
                )
                for column in groups["categorical"]
            ],
            columns=["feature", "most common", "share of the group"],
        ).set_index("feature")
    )

st.header("Groups within the refusal cohort")
st.caption(
    "Not every woman who stays away does so for the same reason. These groups come from "
    "clustering the patients by their SHAP contributions — by *why* the model puts them "
    "at risk — rather than by raw similarity. Two women with the same risk land in "
    "different groups if one lacks a gynaecologist and the other has no contact with any "
    "doctor at all, which is what makes the groups actionable."
)

controls_left, controls_right = st.columns(2)
cluster_count = controls_left.slider("Number of groups", 2, 8, 4)
sample_size = controls_right.select_slider(
    "Patients sampled", options=[2_000, 5_000, 10_000, 20_000], value=5_000,
    help="SHAP is computed per patient, so a sample keeps the page responsive",
)

if st.button("Find the groups"):
    with st.spinner("Computing SHAP values and clustering..."):
        st.session_state["clusters"] = shap_clusters(
            run, refusers, clusters=cluster_count, sample=sample_size
        )

found = st.session_state.get("clusters")
if found is None:
    st.info("Press **Find the groups**.")
else:
    points, contributions = found
    chart, summary = st.columns([2, 3])

    with chart:
        st.pyplot(cluster_figure(points))
        st.caption(
            "Each dot is one patient, placed by the two strongest directions of variation "
            "in SHAP space. The axes have no units — only the grouping means anything."
        )

    with summary:
        for description in cluster_descriptions(points, contributions, refusers):
            with st.container(border=True):
                st.markdown(description)
