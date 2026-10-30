use crate::{error::*, models::*};
use axum::{extract::State, routing::{get, post}, Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{path::PathBuf, sync::Arc};

const PERF_LOW: f64 = 136.86;
const PERF_HIGH: f64 = 241.53;
const INFLUENCERS: [(&str, f32); 4] = [("Macro", 0.), ("Mega", 1.), ("Micro", 2.), ("Nano", 3.)];

pub struct Sujet3 {
    stats_dir: PathBuf,
    config: Map<String, Value>,
    pub regression: ModelGroup,
}

impl Sujet3 {
    pub fn load() -> Self {
        let base = env_path("S3_BASE_DIR", || PathBuf::from("IA/Sujet_3"));
        let models_dir = env_path("S3_MODELS_DIR", || base.join("models"));
        let stats_dir = env_path("S3_STATS_DIR", || base.join("models_stats"));
        let config = load_config(&base.join("models_config.json"));
        Self { regression: load_group(&models_dir, config.get("regression")), stats_dir, config }
    }
}

pub fn router() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/models", get(models))
        .route("/stats", get(stats))
        .route("/predict", post(predict))
}

async fn models(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s3;
    let labels = labels_of(&s.config);
    let list: Vec<Value> = s
        .regression
        .keys()
        .into_iter()
        .map(|k| json!({ "name": k, "label": labels.get(&k).cloned().unwrap_or_else(|| json!(k)) }))
        .collect();
    Ok(Json(json!({ "models": list })))
}

async fn stats(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s3;
    let result: Map<String, Value> = ["regression", "classification"]
        .iter()
        .map(|t| (t.to_string(), stats_map(&s.stats_dir.join(t))))
        .collect();
    Ok(Json(json!({ "stats": result })))
}

fn default_models() -> Vec<String> {
    vec!["linear_regression".into()]
}

#[derive(Deserialize)]
struct MarketingData {
    tv: f64,
    radio: f64,
    social_media: f64,
    influencer: String,
    #[serde(default = "default_models")]
    models: Vec<String>,
}

async fn predict(State(st): State<Arc<crate::AppState>>, Json(d): Json<MarketingData>) -> ApiResult {
    let s = &st.s3;
    let influencer = INFLUENCERS
        .iter()
        .find(|(k, _)| *k == d.influencer)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let allowed: Vec<_> = INFLUENCERS.iter().map(|(k, _)| *k).collect();
            ApiError::unprocessable(format!("Influenceur invalide : {}. Valeurs : {allowed:?}", d.influencer))
        })?;
    let names = validate_models(&d.models, &s.regression)?;

    let total = d.tv + d.radio + d.social_media;
    let share = |v: f64| if total > 0.0 { v / total } else { 0.0 };
    // Ordre = IA/Sujet_3/src/preprocessing.py::FEATURES
    let x: Vec<f32> = [
        d.tv, d.radio, d.social_media, influencer as f64, total,
        share(d.tv), share(d.radio), share(d.social_media), d.tv * d.social_media,
    ]
    .iter()
    .map(|v| *v as f32)
    .collect();

    let mut results = Map::new();
    for name in &names {
        let sales = s.regression.get(name).unwrap().predict_value(&x)? as f64;
        let perf = if sales < PERF_LOW { "Low" } else if sales < PERF_HIGH { "Medium" } else { "High" };
        let roi = if total > 0.0 { json!(round4(sales / total)) } else { Value::Null };
        results.insert(
            name.clone(),
            json!({ "sales_prediction": round4(sales), "performance": perf, "roi_estimate": roi }),
        );
    }
    Ok(Json(json!({ "results": results })))
}
