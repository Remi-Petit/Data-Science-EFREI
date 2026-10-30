use crate::{error::*, models::*};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::{path::PathBuf, sync::Arc};

pub struct Sujet1 {
    base: PathBuf,
    stats_dir: PathBuf,
    config: Map<String, Value>,
    pub failure_24h: ModelGroup,
    pub failure_type: ModelGroup,
    pub rul: ModelGroup,
}

impl Sujet1 {
    pub fn load() -> Self {
        let base = env_path("S1_BASE_DIR", || PathBuf::from("IA/Sujet_1"));
        let models_dir = env_path("S1_MODELS_DIR", || base.join("models"));
        let stats_dir = env_path("S1_STATS_DIR", || base.join("models_stats"));
        let config = load_config(&base.join("models_config.json"));
        Self {
            failure_24h: load_group(&models_dir, config.get("failure_24h")),
            failure_type: load_group(&models_dir, config.get("failure_type")),
            rul: load_group(&models_dir, config.get("rul")),
            base,
            stats_dir,
            config,
        }
    }
}

const TASKS: [&str; 3] = ["failure_24h", "failure_type", "rul"];

pub fn router() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/models", get(models))
        .route("/stats", get(stats))
        .route("/cause-labels", get(cause_labels))
        .route("/predict", post(predict))
}

async fn models(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s1;
    let mut result = Map::new();
    for (group, entries) in &s.config {
        let Some(entries) = entries.as_object() else {
            continue;
        };
        let rank_col = if group == "rul" { "R²" } else { "F1-score" };
        let stats = read_json_dir(&s.stats_dir.join(group));
        let score = |name: &str| {
            stats
                .iter()
                .find(|(k, _)| k == name)
                .and_then(|(_, v)| v.get(rank_col))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        };
        let mut list: Vec<(f64, Value)> = entries
            .iter()
            .map(|(k, v)| {
                let label = v.get("label").cloned().unwrap_or_else(|| json!(k));
                (score(k), json!({ "name": k, "label": label }))
            })
            .collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        result.insert(
            group.clone(),
            Value::Array(list.into_iter().map(|x| x.1).collect()),
        );
    }
    Ok(Json(json!({ "models": result })))
}

async fn stats(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s1;
    let result: Map<String, Value> = TASKS
        .iter()
        .map(|t| (t.to_string(), stats_map(&s.stats_dir.join(t))))
        .collect();
    Ok(Json(json!({ "stats": result })))
}

async fn cause_labels(State(st): State<Arc<crate::AppState>>) -> ApiResult {
    let s = &st.s1;
    let file = s.base.join("labels_causes.json");
    if let Some(labels) = std::fs::read_to_string(file)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
    {
        return Ok(Json(json!({ "labels": labels })));
    }
    let mut labels = Map::new();
    for key in s.failure_type.keys() {
        for c in &s.failure_type.get(&key).unwrap().classes {
            labels.insert(c.clone(), json!(c));
        }
    }
    Ok(Json(json!({ "labels": labels })))
}

fn default_models() -> Vec<String> {
    vec!["random_forest".into()]
}

#[derive(Deserialize)]
struct MachineData {
    vibration_rms: f64,
    temperature_motor: f64,
    current_phase_avg: f64,
    pressure_level: f64,
    rpm: f64,
    hours_since_maintenance: f64,
    ambient_temp: f64,
    machine_type_enc: i64,
    operating_mode_enc: i64,
    hour: i64,
    dayofweek: i64,
    month: i64,
    #[serde(default = "default_models")]
    models: Vec<String>,
    model_cause: Option<String>,
    model_rul: Option<String>,
}

async fn predict(State(st): State<Arc<crate::AppState>>, Json(d): Json<MachineData>) -> ApiResult {
    let s = &st.s1;
    let names = validate_models(&d.models, &s.failure_24h)?;
    let x: Vec<f32> = [
        d.vibration_rms,
        d.temperature_motor,
        d.current_phase_avg,
        d.pressure_level,
        d.rpm,
        d.hours_since_maintenance,
        d.ambient_temp,
        d.machine_type_enc as f64,
        d.operating_mode_enc as f64,
        d.hour as f64,
        d.dayofweek as f64,
        d.month as f64,
    ]
    .iter()
    .map(|v| *v as f32)
    .collect();

    let mut results = Map::new();
    for name in &names {
        let (prediction, p1) = s.failure_24h.get(name).unwrap().predict_binary(&x)?;
        let mut r = Map::new();
        r.insert("prediction".into(), json!(prediction));
        r.insert(
            "label".into(),
            json!(if prediction == 1 {
                "Panne probable"
            } else {
                "Pas de panne"
            }),
        );
        r.insert("probabilite_panne".into(), json!(round4(p1)));

        if prediction == 1 {
            let cause_key = d.model_cause.as_deref().unwrap_or(name);
            if let Some(m) = s
                .failure_type
                .get(cause_key)
                .or_else(|| s.failure_type.get(name))
            {
                let probs = m.predict_proba(&x)?;
                let all: Vec<(&String, f64)> = m
                    .classes
                    .iter()
                    .zip(probs.iter().map(|p| round4(*p as f64)))
                    .collect();
                let mut best: Option<(&String, f64)> = None;
                for (c, p) in all.iter().filter(|(c, _)| c.as_str() != "none") {
                    if best.is_none_or(|(_, bp)| *p > bp) {
                        best = Some((c, *p));
                    }
                }
                if let Some((c, _)) = best {
                    r.insert("cause_potentielle".into(), json!(c));
                }
                let map: Map<String, Value> =
                    all.iter().map(|(c, p)| ((*c).clone(), json!(p))).collect();
                r.insert("probabilites_causes".into(), Value::Object(map));
            }
        }

        let rul_key = d.model_rul.as_deref().unwrap_or(name);
        if let Some(m) = s.rul.get(rul_key).or_else(|| s.rul.get(name)) {
            let rul = m.predict_value(&x)? as f64;
            r.insert(
                "rul_heures".into(),
                json!((rul.max(0.0) * 10.0).round() / 10.0),
            );
        }
        results.insert(name.clone(), Value::Object(r));
    }
    Ok(Json(json!({ "results": results })))
}
