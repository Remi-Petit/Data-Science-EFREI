use crate::error::ApiError;
use ort::{session::Session, value::Tensor};
use serde_json::{Map, Value};
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

pub struct OnnxModel {
    session: Mutex<Session>,
    pub classes: Vec<String>,
}

impl OnnxModel {
    fn load(path: &Path, classes: Vec<String>) -> Result<Self, String> {
        let env = ENV.get_or_init(|| ort::init().build().expect("init ONNX Runtime"));
        let session = Session::builder(env)
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            session: Mutex::new(session),
            classes,
        })
    }

    fn run(&self, features: &[f32], output_idx: usize) -> Result<Vec<f32>, ApiError> {
        let input = Tensor::from_array(([1usize, features.len()], features.to_vec()))
            .map_err(ApiError::internal)?;
        let mut session = self.session.lock().map_err(ApiError::internal)?;
        let outputs = session
            .run(ort::inputs!["X" => input])
            .map_err(ApiError::internal)?;
        let (_, data) = outputs[output_idx]
            .try_extract_tensor::<f32>()
            .map_err(ApiError::internal)?;
        Ok(data.to_vec())
    }

    /// Classifieur binaire : (classe prédite = sortie 0, probabilité de la classe 1 = sortie 1).
    pub fn predict_binary(&self, features: &[f32]) -> Result<(i64, f64), ApiError> {
        let input = Tensor::from_array(([1usize, features.len()], features.to_vec()))
            .map_err(ApiError::internal)?;
        let mut session = self.session.lock().map_err(ApiError::internal)?;
        let outputs = session
            .run(ort::inputs!["X" => input])
            .map_err(ApiError::internal)?;
        let (_, label) = outputs[0]
            .try_extract_tensor::<i64>()
            .map_err(ApiError::internal)?;
        let (_, proba) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(ApiError::internal)?;
        match (label.first(), proba.get(1)) {
            (Some(l), Some(p)) => Ok((*l, f64::from(*p))),
            _ => Err(ApiError::internal("sortie ONNX invalide")),
        }
    }

    /// Classifieur : sortie 1 = probabilités par classe.
    pub fn predict_proba(&self, features: &[f32]) -> Result<Vec<f32>, ApiError> {
        self.run(features, 1)
    }

    /// Régresseur : sortie 0 = valeur prédite.
    pub fn predict_value(&self, features: &[f32]) -> Result<f32, ApiError> {
        self.run(features, 0)?
            .first()
            .copied()
            .ok_or_else(|| ApiError::internal("sortie ONNX vide"))
    }
}

#[derive(Default)]
pub struct ModelGroup(Vec<(String, OnnxModel)>);

impl ModelGroup {
    pub fn get(&self, key: &str) -> Option<&OnnxModel> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, m)| m)
    }

    pub fn keys(&self) -> Vec<String> {
        self.0.iter().map(|(k, _)| k.clone()).collect()
    }
}

pub fn env_path(var: &str, default: impl FnOnce() -> PathBuf) -> PathBuf {
    env::var(var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| default())
}

pub fn load_config(path: &Path) -> Map<String, Value> {
    match fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    {
        Some(Value::Object(m)) => m,
        _ => {
            eprintln!("[models] Config absente ou invalide : {}", path.display());
            Map::new()
        }
    }
}

/// Charge un groupe {clé: {file, label} | filename}; les .joblib sont remplacés par .onnx.
pub fn load_group(models_dir: &Path, group: Option<&Value>) -> ModelGroup {
    let mut models = Vec::new();
    let Some(Value::Object(entries)) = group else {
        return ModelGroup(models);
    };
    for (key, entry) in entries {
        let filename = match entry {
            Value::Object(o) => o.get("file").and_then(Value::as_str),
            other => other.as_str(),
        };
        let Some(filename) = filename else { continue };
        let stem = filename.strip_suffix(".joblib").unwrap_or(filename);
        let path = models_dir.join(format!("{stem}.onnx"));
        if !path.is_file() {
            eprintln!("[models] Modèle absent, ignoré : {}", path.display());
            continue;
        }
        let classes = fs::read_to_string(models_dir.join(format!("{stem}.classes.json")))
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        match OnnxModel::load(&path, classes) {
            Ok(m) => models.push((key.clone(), m)),
            Err(e) => eprintln!("[models] Chargement impossible {} : {e}", path.display()),
        }
    }
    ModelGroup(models)
}

/// Lit tous les *.json d'un dossier, triés par nom : {nom_sans_extension: contenu}.
pub fn read_json_dir(dir: &Path) -> Vec<(String, Value)> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<_> = rd
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort_by_key(|e| e.file_name());
    files
        .into_iter()
        .filter_map(|e| {
            let v = serde_json::from_str(&fs::read_to_string(e.path()).ok()?).ok()?;
            let name = e.path().file_stem()?.to_string_lossy().into_owned();
            Some((name, v))
        })
        .collect()
}

pub fn stats_map(dir: &Path) -> Value {
    Value::Object(read_json_dir(dir).into_iter().collect())
}

pub fn labels_of(config: &Map<String, Value>) -> Map<String, Value> {
    let mut labels = Map::new();
    for group in config.values().filter_map(Value::as_object) {
        for (key, entry) in group {
            if let Some(label) = entry.get("label") {
                labels.insert(key.clone(), label.clone());
            }
        }
    }
    labels
}

pub fn validate_models(requested: &[String], group: &ModelGroup) -> Result<Vec<String>, ApiError> {
    if requested.is_empty() {
        return Err(ApiError::unprocessable(
            "La liste 'models' ne peut pas être vide.",
        ));
    }
    let mut unique: Vec<String> = Vec::new();
    for m in requested {
        if !unique.contains(m) {
            unique.push(m.clone());
        }
    }
    let unavailable: Vec<_> = unique.iter().filter(|m| group.get(m).is_none()).collect();
    if !unavailable.is_empty() {
        return Err(ApiError::unprocessable(format!(
            "Modèle(s) non disponible(s) : {unavailable:?}. Disponibles : {:?}",
            group.keys()
        )));
    }
    Ok(unique)
}

pub fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
