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
    models_info = {
        "sujet_1_models": list(S1_MODELS.keys()),
        "sujet_2_models": list(S2_MODELS.keys()),
        "sujet_3_reg_models": list(S3_REG_MODELS.keys()),
        "sujet_3_cls_models": list(S3_CLS_MODELS.keys()),
    }
    if any(len(v) == 0 for v in models_info.values()):
        return None  # déclenche le 503
    return models_info


app.add_api_route("/health", health([check_models]))