from pydantic import BaseModel, field_validator
from typing import List

S3_INFLUENCER_MAP = {'Macro': 0, 'Mega': 1, 'Micro': 2, 'Nano': 3}
S3_PERF_LOW  = 136.86
S3_PERF_HIGH = 241.53


class MarketingData(BaseModel):
    tv:           float
    radio:        float
    social_media: float
    influencer:   str
    models: List[str] = ["linear_regression"]

    @field_validator("influencer")
    @classmethod
    def influencer_valid(cls, v):
        if v not in S3_INFLUENCER_MAP:
            raise ValueError(f"Influenceur invalide : {v}. Valeurs : {list(S3_INFLUENCER_MAP.keys())}")
        return v

    @field_validator("models")
    @classmethod
    def models_valid(cls, v):
        from ml_models import S3_REG_MODELS
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        v = list(dict.fromkeys(v))
        unavailable = [m for m in v if m not in S3_REG_MODELS]
        if unavailable:
            raise ValueError(f"Modèle(s) non disponible(s) : {unavailable}. Disponibles : {list(S3_REG_MODELS.keys())}")
        return v
