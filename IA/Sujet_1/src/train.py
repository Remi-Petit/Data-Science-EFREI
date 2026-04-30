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

MODEL_FILENAME_TYPE = 'random_forest_failure_type.joblib'


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


def train_and_save(models: dict, X_train, y_train, model_dir: str = 'models') -> dict:
    """Entraîne chaque pipeline, le sauvegarde en joblib et retourne les pipelines entraînés."""
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        out_path = f"{model_dir}/{MODEL_FILENAMES[name]}"
        joblib.dump(pipe, out_path)
        print(f"  {name:25s} → {out_path}")
    return trained


def train_and_save_type(X_train, y_train, model_dir: str = 'models') -> Pipeline:
    """Entraîne un Random Forest multiclasses sur failure_type et le sauvegarde."""
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    out_path = f"{model_dir}/{MODEL_FILENAME_TYPE}"
    joblib.dump(pipe, out_path)
    print(f"  {'Random Forest (failure_type)':25s} → {out_path}")
    return pipe
