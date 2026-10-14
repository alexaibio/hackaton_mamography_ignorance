import pandas as pd
from fastapi import FastAPI

from mammography.api.schemas import PatientScore, ScoreRequest, ScoreResponse
from mammography.model.predict import score_cohort

app = FastAPI(title="Mammography ignorance scoring")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    features = pd.DataFrame([patient.model_dump() for patient in request.patients])
    scored = score_cohort(features)

    return ScoreResponse(
        scores=[
            PatientScore(
                patient_id=row["patient_id"],
                risk=row["risk"],
                top_driver=row.get("top_driver"),
            )
            for row in scored.to_dict("records")
        ]
    )
