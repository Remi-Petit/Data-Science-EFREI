from pydantic import BaseModel, field_validator
from typing import List, Literal

S3RegModelName = Literal["linear_regression", "random_forest", "xgboost", "mlp"]

S3_INFLUENCER_MAP = {'Macro': 0, 'Mega': 1, 'Micro': 2, 'Nano': 3}
S3_PERF_LOW  = 136.86
S3_PERF_HIGH = 241.53


class MarketingData(BaseModel):
    tv:           float
    radio:        float
    social_media: float
    influencer:   Literal["Macro", "Mega", "Micro", "Nano"]
    models: List[S3RegModelName] = ["linear_regression"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))
