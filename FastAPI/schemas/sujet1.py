from pydantic import BaseModel, field_validator
from typing import List


class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    machine_type_enc: int
    operating_mode_enc: int
    hour: int
    dayofweek: int
    month: int
    models: List[str] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_valid(cls, v):
        from ml_models import S1_MODELS
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        v = list(dict.fromkeys(v))
        unavailable = [m for m in v if m not in S1_MODELS]
        if unavailable:
            raise ValueError(f"Modèle(s) non disponible(s) : {unavailable}. Disponibles : {list(S1_MODELS.keys())}")
        return v
