from pydantic import BaseModel, field_validator
from typing import List, Literal

S1ModelName = Literal["logistic_regression", "random_forest", "xgboost"]


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
    models: List[S1ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))
