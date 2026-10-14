from pydantic import BaseModel


class PatientFeatures(BaseModel):
    patient_id: str
    age: int
    region: str
    notification_type: str
    notification_channel: str
    n_invitations: int
    n_prior_invitations: int
    n_prior_attended: int


class ScoreRequest(BaseModel):
    patients: list[PatientFeatures]


class PatientScore(BaseModel):
    patient_id: str
    risk: float
    top_driver: str | None = None


class ScoreResponse(BaseModel):
    scores: list[PatientScore]
