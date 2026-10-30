mod error;
mod models;
mod sujet1;
mod sujet2;
mod sujet3;

use axum::{extract::State, routing::get, Json, Router};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tower_http::{
    compression::CompressionLayer,
    services::{ServeDir, ServeFile},
};

pub struct AppState {
    pub s1: sujet1::Sujet1,
    pub s2: sujet2::Sujet2,
    pub s3: sujet3::Sujet3,
}

impl AppState {
    fn evict_idle(&self, ttl: Duration) {
        let freed = self.s1.failure_24h.evict_idle(ttl)
            + self.s1.failure_type.evict_idle(ttl)
            + self.s1.rul.evict_idle(ttl)
            + self.s2.churn.evict_idle(ttl)
            + self.s3.regression.evict_idle(ttl);
        if freed > 0 {
            models::release_freed_memory();
        }
    }
}

async fn root() -> Json<Value> {
    Json(json!({ "message": "Bienvenue sur l'API Data Science EFREI." }))
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "sujet_1_models": st.s1.failure_24h.keys(),
        "sujet_2_models": st.s2.churn.keys(),
        "sujet_3_reg_models": st.s3.regression.keys(),
    }))
}

async fn api_not_found() -> error::ApiError {
    error::ApiError(axum::http::StatusCode::NOT_FOUND, "Not Found".into())
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        s1: sujet1::Sujet1::load(),
        s2: sujet2::Sujet2::load(),
        s3: sujet3::Sujet3::load(),
    });

    // MODEL_IDLE_SECS=0 (défaut) : aucun modèle n'est gardé en RAM, chargé puis libéré à chaque requête.
    // >0 : les modèles restent en cache ce nombre de secondes d'inactivité (plus rapide, plus de RAM).
    let ttl = Duration::from_secs(
        std::env::var("MODEL_IDLE_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    );
    models::set_idle_ttl(ttl);
    if !ttl.is_zero() {
        let janitor = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30));
            loop {
                tick.tick().await;
                janitor.evict_idle(ttl);
            }
        });
    }

    let api = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .nest("/sujet-1", sujet1::router())
        .nest("/sujet-2", sujet2::router())
        .nest("/sujet-3", sujet3::router())
        .fallback(api_not_found);

    // Frontend Svelte (SPA) : tout ce qui n'est pas un fichier ni /api renvoie index.html.
    let static_dir = std::env::var("STATIC_DIR").unwrap_or_else(|_| "static".into());
    let spa =
        ServeDir::new(&static_dir).fallback(ServeFile::new(format!("{static_dir}/index.html")));

    let app = Router::new()
        .nest("/api", api)
        .fallback_service(spa)
        .layer(CompressionLayer::new())
        .with_state(state);

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8000".into());
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    println!("[backend] écoute sur {addr}");
    axum::serve(listener, app).await.expect("serve");
}
