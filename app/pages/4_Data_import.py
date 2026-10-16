import pandas as pd
import streamlit as st

from mammography.config import get_config, get_settings
from mammography.fhir.client import fetch_resources, search_params
from mammography.fhir.flatten import flatten_resources
from mammography.io import table_path, write_table

RESOURCES = ("Patient", "Communication", "Appointment")

st.set_page_config(page_title="Data import", layout="wide")

st.title("Data import")

settings = get_settings()
config = get_config()
source = st.radio("Source", ["FHIR server", "CSV file"], horizontal=True)

if source == "FHIR server":
    st.subheader("Server")
    st.write(f"`{settings.fhir_base_url or 'FHIR_BASE_URL is not set in .env'}`")

    fetch_config = config.get("fetch", {})
    patient_filter = fetch_config.get("patient", {})
    history_years = fetch_config.get("history_years")

    st.subheader("Download filter")
    st.caption("From config.yaml — a coarse filter, not the training cohort.")
    st.table(
        pd.DataFrame(
            [
                ("Gender", patient_filter.get("gender", "all")),
                ("Minimum age", patient_filter.get("min_age", "no limit")),
                (
                    "History depth",
                    f"last {history_years} years" if history_years else "all history",
                ),
            ],
            columns=["Criterion", "Value"],
        ).set_index("Criterion")
    )

    st.caption("Server-side filters that will be applied:")
    st.table(
        pd.DataFrame(
            [
                (
                    resource_type,
                    ", ".join(f"{k}={v}" for k, v in search_params(resource_type).items()),
                )
                for resource_type in RESOURCES
            ],
            columns=["Resource", "FHIR search filter"],
        ).set_index("Resource")
    )

    limit = st.number_input(
        "Stop after N resources per type (0 = no limit)", min_value=0, value=0, step=50
    )

    if st.button("Download and flatten"):
        raw_dir = settings.data_folder / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for resource_type in RESOURCES:
            with st.spinner(f"Fetching {resource_type}..."):
                written = fetch_resources(
                    resource_type,
                    raw_dir / f"{resource_type}.ndjson",
                    limit=int(limit) or None,
                )
            st.write(f"{resource_type}: {written} resources")

        with st.spinner("Flattening..."):
            flatten_resources()
        st.success("Written to data/interim/")
        st.cache_data.clear()

else:
    st.subheader("Upload a flattened patients table")
    st.caption(
        "Expects the same columns `mammo flatten` produces: patient_id, birth_date, "
        "gender, deceased_date, city, district, state, postal_code, country, "
        "marital_status, preferred_language, has_phone, has_mobile, has_email, "
        "has_registered_gp."
    )

    upload = st.file_uploader("CSV file", type="csv")

    if upload is not None:
        uploaded = pd.read_csv(upload)
        st.write(f"{len(uploaded)} rows, {len(uploaded.columns)} columns")
        st.dataframe(uploaded.head(20), width="stretch")

        destination = table_path("interim", "patients")
        st.warning(f"Saving overwrites {destination}")
        if st.button("Save as the interim patients table"):
            write_table(uploaded, "interim", "patients")
            st.success(f"Written to {destination}")
            st.cache_data.clear()
