from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Literal

app = FastAPI()

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]

# Chargement des modèles (MODELS_DIR configurable via variable d'environnement)
_models_dir = os.getenv('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_1', 'models'))
MODELS = {
    "logistic_regression": joblib.load(os.path.join(_models_dir, 'logistic_regression_failure_24h.joblib')),
    "random_forest":       joblib.load(os.path.join(_models_dir, 'random_forest_failure_24h.joblib')),
    "xgboost":             joblib.load(os.path.join(_models_dir, 'xgboost_failure_24h.joblib')),
}

# Schéma des données d'entrée
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
    models: List[ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))  # dédoublonnage en conservant l'ordre

@app.get("/health")
def health():
    return {"status": "ok", "models_available": list(MODELS.keys())}

@app.post("/predict")
def predict(data: MachineData):
    features = data.model_dump(exclude={"models"})
    df = pd.DataFrame([features])

    results = {}
    for model_name in data.models:
        model = MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        results[model_name] = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4)
        }

    return {"results": results}