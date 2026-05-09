from fastapi import APIRouter
from schemas.sujet3 import MarketingData
from ml_models import S3_REG_MODELS
import controllers.sujet3 as controller
import os
import json

router = APIRouter(prefix="/sujet-3", tags=["Sujet 3 – Marketing ROI"])

_STATS_DIR = os.getenv('S3_STATS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'IA', 'Sujet_3', 'models_stats'))


@router.get("/models")
def get_models():
    return {"models": list(S3_REG_MODELS.keys())}


@router.get("/stats")
def get_stats():
    result = {"regression": {}, "classification": {}}
    for task in ("regression", "classification"):
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
def predict(data: MarketingData):
    return controller.predict(data)
