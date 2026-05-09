from fastapi import APIRouter
from schemas.sujet1 import MachineData
from ml_models import S1_MODELS
import controllers.sujet1 as controller
import os
import json

router = APIRouter(prefix="/sujet-1", tags=["Sujet 1 – Maintenance prédictive"])

_STATS_DIR = os.getenv('S1_STATS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'IA', 'Sujet_1', 'models_stats'))


@router.get("/models")
def get_models():
    return {"models": list(S1_MODELS.keys())}


@router.get("/stats")
def get_stats():
    result = {"failure_24h": {}, "failure_type": {}}
    for task in ("failure_24h", "failure_type"):
        task_dir = os.path.join(_STATS_DIR, task)
        if not os.path.isdir(task_dir):
            continue
        for fname in sorted(os.listdir(task_dir)):
            if fname.endswith('.json'):
                model_key = fname[:-5]
                with open(os.path.join(task_dir, fname), encoding='utf-8') as f:
                    result[task][model_key] = json.load(f)
    return {"stats": result}


@router.post("/predict")
def predict(data: MachineData):
    return controller.predict(data)
