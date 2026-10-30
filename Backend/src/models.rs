use crate::error::ApiError;
use ort::{ep, session::Session, value::Tensor};
use serde_json::{Map, Value};
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
static IDLE_TTL: OnceLock<Duration> = OnceLock::new();
/// En mode sans cache, une seule session existe à la fois : la RAM reste bornée sous charge.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Durée de conservation des sessions chargées ; 0 = déchargées dès la fin de la requête.
pub fn set_idle_ttl(ttl: Duration) {
    let _ = IDLE_TTL.set(ttl);
}

fn idle_ttl() -> Duration {
    IDLE_TTL.get().copied().unwrap_or_default()
}

/// Session ONNX chargée à la demande (certains modèles pèsent plus d'1 Go en RAM une fois chargés).
struct Loaded {
    session: Session,
    last_used: Instant,
}

pub struct OnnxModel {
    path: PathBuf,
    slot: Mutex<Option<Loaded>>,
    pub classes: Vec<String>,
}

fn build_session(path: &Path) -> Result<Session, String> {
    let env = ENV.get_or_init(|| ort::init().build().expect("init ONNX Runtime"));
    let err = |e: &dyn std::fmt::Display| e.to_string();
    // Un thread, pas d'arène ni de pattern mémoire : une inférence = une seule ligne.
    let builder = Session::builder(env).map_err(|e| err(&e))?;
    let builder = builder.with_intra_threads(1).map_err(|e| err(&e))?;
    let builder = builder.with_inter_threads(1).map_err(|e| err(&e))?;
    let builder = builder.with_memory_pattern(false).map_err(|e| err(&e))?;
    let mut builder = builder
        .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(|e| err(&e))?;
    builder.commit_from_file(path).map_err(|e| err(&e))
}

impl OnnxModel {
    fn new(path: PathBuf, classes: Vec<String>) -> Self {
        Self {
            path,
            slot: Mutex::new(None),
            classes,
        }
    }

    fn with_session<T>(
        &self,
        f: impl FnOnce(&mut Session) -> Result<T, ApiError>,
    ) -> Result<T, ApiError> {
        let load = || {
            build_session(&self.path).map_err(|e| {
                ApiError::internal(format!("chargement de {} : {e}", self.path.display()))
            })
        };
        if idle_ttl().is_zero() {
            let _gate = ONE_AT_A_TIME.lock().map_err(ApiError::internal)?;
            let mut session = load()?;
            let result = f(&mut session);
            drop(session);
            release_freed_memory();
            return result;
        }
        let mut slot = self.slot.lock().map_err(ApiError::internal)?;
        if slot.is_none() {
            *slot = Some(Loaded {
                session: load()?,
                last_used: Instant::now(),
            });
        }
        let loaded = slot.as_mut().expect("session chargée");
        loaded.last_used = Instant::now();
        f(&mut loaded.session)
    }

    /// Libère la session si elle n'a pas servi depuis `ttl`.
    fn evict_if_idle(&self, ttl: Duration) -> bool {
        let Ok(mut slot) = self.slot.lock() else {
            return false;
        };
        let idle = slot.as_ref().is_some_and(|l| l.last_used.elapsed() >= ttl);
        if idle {
            *slot = None;
        }
        idle
    }

    /// Exécute le modèle sur `rows` (une seule charge de session pour tout le lot).
    fn run_batch<T>(
        &self,
        rows: &[Vec<f32>],
        extract: impl FnOnce(&ort::session::SessionOutputs<'_>) -> Result<T, ApiError>,
    ) -> Result<T, ApiError> {
        let n_features = rows.first().map_or(0, Vec::len);
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let input =
            Tensor::from_array(([rows.len(), n_features], flat)).map_err(ApiError::internal)?;
        self.with_session(|session| {
            let outputs = session
                .run(ort::inputs!["X" => input])
                .map_err(ApiError::internal)?;
            extract(&outputs)
        })
    }

    /// Classifieur binaire : par ligne, (classe prédite = sortie 0, probabilité de la classe 1 = sortie 1).
    pub fn predict_binary_batch(&self, rows: &[Vec<f32>]) -> Result<Vec<(i64, f64)>, ApiError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.run_batch(rows, |outputs| {
            let (_, label) = outputs[0]
                .try_extract_tensor::<i64>()
                .map_err(ApiError::internal)?;
            let (_, proba) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(ApiError::internal)?;
            let n = rows.len();
            if label.len() != n || proba.len() != n * 2 {
                return Err(ApiError::internal("sortie ONNX invalide"));
            }
            Ok((0..n)
                .map(|i| (label[i], f64::from(proba[2 * i + 1])))
                .collect())
        })
    }

    /// Classifieur : par ligne, probabilités de chaque classe (sortie 1).
    pub fn predict_proba_batch(&self, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, ApiError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.run_batch(rows, |outputs| {
            let (_, proba) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(ApiError::internal)?;
            let n_classes = proba.len() / rows.len();
            Ok(proba
                .chunks(n_classes.max(1))
                .map(<[f32]>::to_vec)
                .collect())
        })
    }

    /// Régresseur : une valeur prédite par ligne (sortie 0).
    pub fn predict_value_batch(&self, rows: &[Vec<f32>]) -> Result<Vec<f32>, ApiError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.run_batch(rows, |outputs| {
            let (_, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(ApiError::internal)?;
            Ok(data.to_vec())
        })
    }

    pub fn predict_binary(&self, features: &[f32]) -> Result<(i64, f64), ApiError> {
        let rows = [features.to_vec()];
        self.predict_binary_batch(&rows)?
            .pop()
            .ok_or_else(|| ApiError::internal("sortie ONNX vide"))
    }

    pub fn predict_value(&self, features: &[f32]) -> Result<f32, ApiError> {
        let rows = [features.to_vec()];
        self.predict_value_batch(&rows)?
            .pop()
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

    /// Décharge les sessions inactives ; renvoie le nombre de modèles libérés.
    pub fn evict_idle(&self, ttl: Duration) -> usize {
        self.0.iter().filter(|(_, m)| m.evict_if_idle(ttl)).count()
    }
}

/// Rend au système la mémoire libérée par les modèles déchargés (glibc la garde sinon).
pub fn release_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: malloc_trim n'a aucune précondition.
        unsafe {
            malloc_trim(0);
        }
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
        models.push((key.clone(), OnnxModel::new(path, classes)));
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
