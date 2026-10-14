import streamlit as st

st.set_page_config(page_title="Not responding patients", layout="wide")

st.title("Patients unlikely to respond")
st.info(
    "Pending the invitation layer and a trained model: needs "
    "data/processed/invitations.csv and a risk score per invitation."
)
