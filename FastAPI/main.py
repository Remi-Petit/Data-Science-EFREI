from fastapi import FastAPI
from fastapi_health import health
from ml_models import S1_MODELS, S2_MODELS, S3_REG_MODELS, S3_CLS_MODELS
from routers import sujet1, sujet2, sujet3

app = FastAPI(root_path="/api")

app.include_router(sujet1.router)
app.include_router(sujet2.router)
app.include_router(sujet3.router)


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Data Science EFREI. La documentation est disponible sur /api/docs."}


def check_models():
    return {
        "sujet_1_models": list(S1_MODELS.keys()),
        "sujet_2_models": list(S2_MODELS.keys()),
        "sujet_3_reg_models": list(S3_REG_MODELS.keys()),
        "sujet_3_cls_models": list(S3_CLS_MODELS.keys()),
    }


app.add_api_route("/health", health([check_models]))