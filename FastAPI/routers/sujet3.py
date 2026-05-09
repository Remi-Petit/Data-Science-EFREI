from fastapi import APIRouter
from schemas.sujet3 import MarketingData
import controllers.sujet3 as controller

router = APIRouter(prefix="/sujet-3", tags=["Sujet 3 – Marketing ROI"])


@router.post("/predict")
def predict(data: MarketingData):
    return controller.predict(data)
