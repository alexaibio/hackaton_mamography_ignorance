from fastapi import FastAPI

from mammography.api.schemas import ScoreRequest, ScoreResponse

app = FastAPI(title="Mammography ignorance scoring")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    raise NotImplementedError("scoring is not implemented yet")
