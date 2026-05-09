from fastapi import APIRouter
from schemas.sujet2 import ChurnData
import controllers.sujet2 as controller

router = APIRouter(prefix="/sujet-2", tags=["Sujet 2 – Churn client"])


@router.post("/predict")
def predict(data: ChurnData):
    return controller.predict(data)
