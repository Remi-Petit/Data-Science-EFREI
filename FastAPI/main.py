from fastapi import FastAPI, HTTPException
import joblib
import os
import sys
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Literal

# Rendre le module src (IA/Sujet_1/src) importable par pickle lors du chargement des modèles
_ia_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'IA', 'Sujet_1')
if _ia_dir not in sys.path:
    sys.path.insert(0, _ia_dir)

app = FastAPI()

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]

# Chargement des modèles (MODELS_DIR configurable via variable d'environnement)
_models_dir = os.getenv('MODELS_DIR', os.path.join(os.path.dirname(__file__), '..', 'IA', 'Sujet_1', 'models'))
MODELS = {
    "logistic_regression": joblib.load(os.path.join(_models_dir, 'logistic_regression_failure_24h.joblib')),
    "random_forest":       joblib.load(os.path.join(_models_dir, 'random_forest_failure_24h.joblib')),
    "xgboost":             joblib.load(os.path.join(_models_dir, 'xgboost_failure_24h.joblib')),
}
MODELS_TYPE = {
    "logistic_regression": joblib.load(os.path.join(_models_dir, 'logistic_regression_failure_type.joblib')),
    "random_forest":       joblib.load(os.path.join(_models_dir, 'random_forest_failure_type.joblib')),
    "xgboost":             joblib.load(os.path.join(_models_dir, 'xgboost_failure_type.joblib')),
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
        result = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4),
        }
        if prediction == 1:
            model_type = MODELS_TYPE[model_name]
            type_proba = model_type.predict_proba(df)[0]
            type_classes = model_type.classes_
            all_scores = {cls: round(float(p), 4) for cls, p in zip(type_classes, type_proba)}
            failure_scores = {cls: p for cls, p in all_scores.items() if cls != 'none'}
            result["cause_potentielle"] = max(failure_scores, key=failure_scores.get)
            result["probabilites_causes"] = all_scores
        results[model_name] = result

    return {"results": results}