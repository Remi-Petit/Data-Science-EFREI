from fastapi import APIRouter
from schemas.sujet1 import MachineData
import controllers.sujet1 as controller

router = APIRouter(prefix="/sujet-1", tags=["Sujet 1 – Maintenance prédictive"])


@router.post("/predict")
def predict(data: MachineData):
    return controller.predict(data)
