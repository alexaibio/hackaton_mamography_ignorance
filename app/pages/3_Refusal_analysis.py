import pandas as pd
import streamlit as st

from mammography.llm.messages import draft_invitation_message

PLACEHOLDER = "Illustrative numbers — replaced once the invitation layer and labels exist."

BY_AGE = pd.DataFrame(
    [("45-49", 61), ("50-54", 54), ("55-59", 47), ("60-64", 44), ("65-69", 52)],
    columns=["age band", "did not respond %"],
).set_index("age band")

BY_CHANNEL = pd.DataFrame(
    [("Letter only", 58), ("Letter + SMS", 39), ("SMS only", 44), ("Phone call", 27)],
    columns=["channel", "did not respond %"],
).set_index("channel")

BY_HISTORY = pd.DataFrame(
    [("First invitation", 42), ("Second", 55), ("Third", 68), ("Fourth or later", 74)],
    columns=["invitation number", "did not respond %"],
).set_index("invitation number")

INTERVENTIONS = pd.DataFrame(
    [
        ("Add SMS reminder 7 days before", "no mobile on file is the blocker", "low", 6),
        ("Phone call by screening centre", "never-responders", "high", 14),
        ("GP endorsement letter", "has a registered GP", "medium", 9),
        ("Letter in preferred language", "non-Czech speakers", "low", 11),
        ("Pre-booked appointment slot", "all segments", "medium", 17),
    ],
    columns=["Intervention", "Target group", "Cost", "Expected uplift pp"],
)

st.set_page_config(page_title="Refusal analysis", layout="wide")

st.title("Refusal analysis")
st.caption("Synthetic demo data — not real screening outcomes.")
st.info(PLACEHOLDER)

st.header("Who does not respond")

left, middle, right = st.columns(3)
with left:
    st.subheader("By age band")
    st.bar_chart(BY_AGE)
with middle:
    st.subheader("By channel used")
    st.bar_chart(BY_CHANNEL)
with right:
    st.subheader("By invitation number")
    st.bar_chart(BY_HISTORY)

st.header("Proposed interventions")
st.table(INTERVENTIONS.set_index("Intervention"))

st.header("A/B test the message")

intervention = st.selectbox("Intervention", INTERVENTIONS["Intervention"].tolist())
channel = st.selectbox("Channel", ["letter", "sms", "email"])
chosen = INTERVENTIONS.loc[INTERVENTIONS["Intervention"] == intervention].iloc[0]

st.write(f"Target group: **{chosen['Target group']}** · expected uplift **{chosen['Expected uplift pp']} pp**")

variant_a, variant_b = st.columns(2)

with variant_a:
    st.subheader("A — current wording")
    st.text_area(
        "a",
        "Zveme Vás na preventivní mammografické vyšetření. Prosíme, ozvěte se na "
        "[TELEFON] a sjednejte si termín.",
        height=220,
        label_visibility="collapsed",
    )

with variant_b:
    st.subheader("B — intervention wording")
    if st.button("Generate variant B"):
        with st.spinner("Drafting..."):
            st.session_state["variant_b"] = draft_invitation_message(
                {
                    "notification_channel": channel,
                    "top_driver": chosen["Target group"],
                    "notification_type": intervention,
                },
                channel,
            )
    st.text_area(
        "b",
        st.session_state.get("variant_b", ""),
        height=220,
        label_visibility="collapsed",
    )
