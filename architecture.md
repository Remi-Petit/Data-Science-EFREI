Project Path: Data

Source Tree:

```txt
Data
├── FastAPI
│   ├── __pycache__
│   │   └── main.cpython-311.pyc
│   └── main.py
├── IA
│   ├── Sujet_1
│   │   ├── data
│   │   │   └── industrial_machine_maintenance.csv
│   │   ├── models
│   │   │   ├── logistic_regression_failure_24h.joblib
│   │   │   ├── random_forest_failure_24h.joblib
│   │   │   └── xgboost_failure_24h.joblib
│   │   ├── notebook
│   │   │   └── maintenance_ml.ipynb
│   │   ├── results
│   │   │   ├── confusion_matrices.png
│   │   │   ├── feature_importance.png
│   │   │   └── roc_curves.png
│   │   ├── src
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── evaluate.cpython-311.pyc
│   │   │   │   ├── preprocessing.cpython-311.pyc
│   │   │   │   └── train.cpython-311.pyc
│   │   │   ├── evaluate.py
│   │   │   ├── preprocessing.py
│   │   │   └── train.py
│   │   └── train_pipeline.py
│   ├── Sujet_2
│   └── Sujet_3
├── README.md
└── Streamlit
    └── main.py

```

`FastAPI\main.py`:

```py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Literal

app = FastAPI()

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]

# Chargement des modèles
MODELS = {
    "logistic_regression": joblib.load('../IA/Sujet_1/logistic_regression_failure_24h.joblib'),
    "random_forest":       joblib.load('../IA/Sujet_1/random_forest_failure_24h.joblib'),
    "xgboost":             joblib.load('../IA/Sujet_1/xgboost_failure_24h.joblib'),
}

# Schéma des données d'entrée
class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    machine_type_enc: int
    operating_mode_enc: int
    hour: int
    dayofweek: int
    month: int
    models: List[ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))  # dédoublonnage en conservant l'ordre

@app.get("/health")
def health():
    return {"status": "ok", "models_available": list(MODELS.keys())}

@app.post("/predict")
def predict(data: MachineData):
    features = data.model_dump(exclude={"models"})
    df = pd.DataFrame([features])

    results = {}
    for model_name in data.models:
        model = MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        results[model_name] = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4)
        }

    return {"results": results}
```

`IA\Sujet_1\data\industrial_machine_maintenance.csv`:

```csv
CSV Schema (1 sample row):
Headers: timestamp, machine_id, machine_type, vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm, operating_mode, hours_since_maintenance, ambient_temp, rul_hours, failure_within_24h, failure_type, estimated_repair_cost
Sample: "2024-01-01 00:00:00", "1", "CNC", "0.81", "49.51", "5.1", "23.6", "860.9", "idle", "273.8", "13.9", "61.0", "0", "none", "0"
... [24041 more rows omitted]

```

`IA\Sujet_1\notebook\maintenance_ml.ipynb`:

```ipynb
Jupyter Notebook Summary:
Total cells: 28 (17 code, 11 markdown, 0 raw)

Code Cell #1:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
print('Librairies EDA chargees OK')
```

Code Cell #2:
```python
df = pd.read_csv('../data/industrial_machine_maintenance.csv', parse_dates=['timestamp'])
print(f'Shape : {df.shape}')
df.head()
```

Code Cell #3:
```python
df.info()
```

... [14 more code cells omitted]

```

`IA\Sujet_1\src\evaluate.py`:

```py
"""
Évaluation des modèles : métriques, matrices de confusion, courbes ROC,
importance des features.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
)


def evaluate_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcule accuracy, precision, recall, F1 et ROC-AUC pour chaque modèle.
    Retourne un DataFrame comparatif.
    """
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        report = classification_report(
            y_test, y_pred,
            target_names=['Pas de panne', 'Panne'],
            output_dict=True
        )
        rows.append({
            'Modèle':     name,
            'Accuracy':   round(report['accuracy'], 4),
            'Precision':  round(report['Panne']['precision'], 4),
            'Recall':     round(report['Panne']['recall'], 4),
            'F1-score':   round(report['Panne']['f1-score'], 4),
            'ROC-AUC':    round(roc_auc_score(y_test, y_prob), 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def plot_confusion_matrices(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Affiche les matrices de confusion côte à côte."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, trained_models.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Pas de panne', 'Panne'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    plt.suptitle('Matrices de confusion (jeu de test)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curves(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Courbes ROC pour tous les modèles sur un même graphe."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    ax.set_title('Courbes ROC – Comparaison des modèles')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(
    trained_models: dict,
    features: list,
    model_name: str = 'Random Forest'
) -> tuple[plt.Figure, pd.Series]:
    """Bar chart de l'importance des features (modèles basés sur les arbres)."""
    pipe = trained_models[model_name]
    importances = pipe.named_steps['clf'].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    feat_imp.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
    ax.set_title(f'Importance des features – {model_name}')
    ax.set_ylabel('Importance')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    return fig, feat_imp

```

`IA\Sujet_1\src\preprocessing.py`:

```py
"""
Preprocessing pipeline – chargement, feature engineering, split train/test.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = 'industrial_machine_maintenance.csv'

FEATURES = [
    'vibration_rms', 'temperature_motor', 'current_phase_avg',
    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp',
    'machine_type_enc', 'operating_mode_enc', 'hour', 'dayofweek', 'month'
]
TARGET = 'failure_within_24h'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path, parse_dates=['timestamp'])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features temporelles et encode les variables catégorielles."""
    df = df.copy()

    # Features temporelles
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Encodage label des variables catégorielles
    le_type = LabelEncoder()
    le_mode = LabelEncoder()
    df['machine_type_enc'] = le_type.fit_transform(df['machine_type'].astype(str))
    df['operating_mode_enc'] = le_mode.fit_transform(df['operating_mode'].astype(str))

    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Retourne X_train, X_test, y_train, y_test (split stratifié)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

```

`IA\Sujet_1\src\train.py`:

```py
"""
Définition des pipelines ML, cross-validation et entraînement final.
"""
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

MODEL_FILENAMES = {
    'Logistic Regression': 'logistic_regression_failure_24h.joblib',
    'Random Forest':       'random_forest_failure_24h.joblib',
    'XGBoost':             'xgboost_failure_24h.joblib',
}


def build_pipelines() -> dict:
    """Retourne un dict {nom: Pipeline sklearn}."""
    return {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                scale_pos_weight=1, random_state=42, n_jobs=-1,
                eval_metric='logloss'
            ))
        ]),
    }


def cross_validate_models(models: dict, X_train, y_train, n_splits: int = 5) -> dict:
    """Lance une StratifiedKFold CV et retourne les scores ROC-AUC par modèle."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = scores
    return results


def train_and_save(models: dict, X_train, y_train, model_dir: str = '.') -> dict:
    """Entraîne chaque pipeline, le sauvegarde en joblib et retourne les pipelines entraînés."""
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        out_path = f"{model_dir}/{MODEL_FILENAMES[name]}"
        joblib.dump(pipe, out_path)
        print(f"  {name:25s} → {out_path}")
    return trained

```

`IA\Sujet_1\train_pipeline.py`:

```py
"""
Pipeline d'entraînement – exécutable en ligne de commande.

Usage (depuis IA/Sujet_1/) :
    python train_pipeline.py
"""
import os
import sys

# Résolution des imports locaux quand on exécute le script directement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, engineer_features, get_train_test_split, FEATURES
from src.train import build_pipelines, cross_validate_models, train_and_save
from src.evaluate import evaluate_models, plot_confusion_matrices, plot_roc_curves, plot_feature_importance


def main():
    # ── 1. Données ────────────────────────────────────────────────────────────
    print("=== Chargement et préparation des données ===")
    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"Train : {X_train.shape[0]} lignes  |  Test : {X_test.shape[0]} lignes")
    print(f"Taux de pannes (train) : {y_train.mean():.2%}")

    # ── 2. Cross-validation ───────────────────────────────────────────────────
    print("\n=== Cross-validation (StratifiedKFold 5 folds) ===")
    models = build_pipelines()
    cv_results = cross_validate_models(models, X_train, y_train)
    for name, scores in cv_results.items():
        print(f"  {name:25s} | ROC-AUC CV = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── 3. Entraînement final + sauvegarde ────────────────────────────────────
    print("\n=== Entraînement final + sauvegarde des modèles ===")
    trained = train_and_save(models, X_train, y_train)

    # ── 4. Évaluation ────────────────────────────────────────────────────────
    print("\n=== Évaluation sur le jeu de test ===")
    results_df = evaluate_models(trained, X_test, y_test)
    print(results_df.to_string())

    # ── 5. Visualisations (sauvegardées en PNG) ───────────────────────────────
    fig_cm = plot_confusion_matrices(trained, X_test, y_test)
    fig_cm.savefig('confusion_matrices.png', dpi=120)

    fig_roc = plot_roc_curves(trained, X_test, y_test)
    fig_roc.savefig('roc_curves.png', dpi=120)

    fig_fi, _ = plot_feature_importance(trained, FEATURES)
    fig_fi.savefig('feature_importance.png', dpi=120)

    print("\nFigures sauvegardées : confusion_matrices.png, roc_curves.png, feature_importance.png")
    print("\nPipeline terminé ✓")


if __name__ == '__main__':
    main()

```

`README.md`:

```md
# Code2Prompt
code2prompt . --output-file architecture.md
```

`Streamlit\main.py`:

```py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}

st.title("🔧 Prédiction de panne machine")
st.markdown("Renseignez les paramètres de la machine pour prédire un risque de panne dans les 24h.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    vibration_rms = st.number_input("Vibration RMS", value=2.0, step=0.1)
    temperature_motor = st.number_input("Température moteur (°C)", value=70.0, step=0.5)
    current_phase_avg = st.number_input("Courant moyen (A)", value=10.0, step=0.1)
    ambient_temp = st.number_input("Température ambiante (°C)", value=25.0, step=0.5)

with col2:
    pressure_level = st.number_input("Pression (bar)", value=3.0, step=0.1)
    rpm = st.number_input("RPM", value=1450.0, step=10.0)
    hours_since_maintenance = st.number_input("Heures depuis maintenance", value=200.0, step=10.0)

with col3:
    machine_type_enc = st.selectbox("Type de machine", options=[0, 1, 2], format_func=lambda x: ["Pump", "Compressor", "Motor"][x])
    operating_mode_enc = st.selectbox("Mode opératoire", options=[0, 1, 2], format_func=lambda x: ["normal", "idle", "peak"][x])
    hour = st.slider("Heure", 0, 23, 12)
    dayofweek = st.slider("Jour de la semaine", 0, 6, 1)
    month = st.slider("Mois", 1, 12, 6)

st.divider()

selected_models = st.multiselect(
    "🤖 Modèles à comparer",
    options=list(MODEL_LABELS.keys()),
    default=["random_forest"],
    format_func=lambda x: MODEL_LABELS[x],
)

st.divider()

if st.button("🔍 Lancer la prédiction", use_container_width=True):
    if not selected_models:
        st.warning("Veuillez sélectionner au moins un modèle.")
        st.stop()

    payload = {
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp,
        "machine_type_enc": machine_type_enc,
        "operating_mode_enc": operating_mode_enc,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "models": selected_models,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        st.divider()
        st.subheader("📊 Résultats par modèle")

        cols = st.columns(len(results))
        for col, (model_key, res) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {MODEL_LABELS[model_key]}")
                if res["prediction"] == 1:
                    st.error(f"⚠️ {res['label']}")
                else:
                    st.success(f"✅ {res['label']}")
                st.metric("Probabilité de panne", f"{res['probabilite_panne'] * 100:.2f}%")

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")

```