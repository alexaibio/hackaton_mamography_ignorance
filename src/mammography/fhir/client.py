import json
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from mammography.config import get_config, get_settings

PAGE_SIZE = 100
TIMEOUT = 30.0
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _next_link(bundle: dict) -> str | None:
    for link in bundle.get("link", []):
        if link.get("relation") == "next":
            return link.get("url")
    return None


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Accept": "application/fhir+json"}
    if settings.fhir_token:
        headers["Authorization"] = f"Bearer {settings.fhir_token}"
    return headers


HISTORY_DATE_PARAM = {
    "Communication": "sent",
    "Appointment": "date",
    "Encounter": "date",
    "Procedure": "date",
    "Observation": "date",
}


def search_params(resource_type: str) -> dict[str, str]:
    fetch_config = get_config().get("fetch", {})
    params = {}

    patient_filter = fetch_config.get("patient", {})
    if resource_type == "Patient":
        if "gender" in patient_filter:
            params["gender"] = patient_filter["gender"]
        if "min_age" in patient_filter:
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=patient_filter["min_age"])
            params["birthdate"] = f"le{cutoff.date().isoformat()}"

    history_years = fetch_config.get("history_years")
    date_param = HISTORY_DATE_PARAM.get(resource_type)
    if history_years and date_param:
        since = pd.Timestamp.today() - pd.DateOffset(years=history_years)
        params[date_param] = f"ge{since.date().isoformat()}"

    return params


def fetch_resources(resource_type: str, out_path: Path, limit: int | None = None) -> int:
    settings = get_settings()
    url = f"{settings.fhir_base_url}/{resource_type}"
    params: dict | None = {"_count": PAGE_SIZE, **search_params(resource_type)}
    written = 0

    with httpx.Client(headers=_headers(), timeout=TIMEOUT) as client:
        with out_path.open("w") as f:
            while url:
                bundle = _get(client, url, params)
                params = None
                for entry in bundle.get("entry", []):
                    f.write(json.dumps(entry["resource"]) + "\n")
                    written += 1
                    if limit is not None and written >= limit:
                        return written
                url = _next_link(bundle)

    return written
