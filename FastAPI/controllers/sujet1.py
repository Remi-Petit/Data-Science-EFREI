import pandas as pd
from schemas.sujet1 import MachineData
from ml_models import S1_MODELS, S1_MODELS_TYPE, S1_MODELS_RUL


def predict(data: MachineData) -> dict:
    features = data.model_dump(exclude={"models"})
    df = pd.DataFrame([features])

    results = {}
    for model_name in data.models:
        model = S1_MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        result = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4),
        }
        if prediction == 1:
            model_type = S1_MODELS_TYPE[model_name]
            type_proba = model_type.predict_proba(df)[0]
            type_classes = model_type.classes_
            all_scores = {cls: round(float(p), 4) for cls, p in zip(type_classes, type_proba)}
            failure_scores = {cls: p for cls, p in all_scores.items() if cls != 'none'}
            result["cause_potentielle"] = max(failure_scores, key=failure_scores.get)
            result["probabilites_causes"] = all_scores
        if model_name in S1_MODELS_RUL:
            rul_model = S1_MODELS_RUL[model_name]
            rul = float(rul_model.predict(df)[0])
            result["rul_heures"] = round(max(rul, 0.0), 1)
        results[model_name] = result

    return {"results": results}
