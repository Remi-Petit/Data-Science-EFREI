use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub struct ApiError(pub StatusCode, pub String);

impl ApiError {
    pub fn unprocessable(msg: impl Into<String>) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, msg.into())
    }

    pub fn internal(msg: impl ToString) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "detail": self.1 }))).into_response()
    }
}

pub type ApiResult = Result<Json<serde_json::Value>, ApiError>;
