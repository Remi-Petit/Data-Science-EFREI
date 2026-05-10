"""
Définition des pipelines ML, cross-validation et entraînement final.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier, XGBRegressor

MODEL_FILENAMES = {
    'Logistic Regression': 'logistic_regression_failure_24h.joblib',
    'Random Forest':       'random_forest_failure_24h.joblib',
    'XGBoost':             'xgboost_failure_24h.joblib',
    'MLP':                 'mlp_failure_24h.joblib',
    'SVM':                 'svm_failure_24h.joblib',
}

MODEL_FILENAMES_TYPE = {
    'Logistic Regression': 'logistic_regression_failure_type.joblib',
    'Random Forest':       'random_forest_failure_type.joblib',
    'XGBoost':             'xgboost_failure_type.joblib',
    'MLP':                 'mlp_failure_type.joblib',
    'SVM':                 'svm_failure_type.joblib',
}


class XGBClassifierWithEncoder(BaseEstimator, ClassifierMixin):
    """Wrapper autour de XGBClassifier qui encode les labels string en entiers."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.le_ = LabelEncoder()
        self.xgb_ = XGBClassifier(**kwargs)

    def fit(self, X, y):
        y_enc = self.le_.fit_transform(y)
        self.xgb_.fit(X, y_enc)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        y_enc = self.xgb_.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.xgb_.predict_proba(X)


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
        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu',
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', C=1.0, class_weight='balanced',
                random_state=42, probability=True
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


def build_pipelines_type() -> dict:
    """Retourne les pipelines pour la classification failure_type (multiclasses)."""
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
            ('clf', XGBClassifierWithEncoder(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=42, n_jobs=-1, eval_metric='mlogloss',
            ))
        ]),
        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu',
                max_iter=500, random_state=42, early_stopping=False,
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', C=1.0, class_weight='balanced',
                random_state=42, probability=True
            ))
        ]),
    }


def train_and_save_type(X_train, y_train, model_dir: str = 'models') -> dict:
    """Entraîne les 3 pipelines multiclasses sur failure_type et les sauvegarde."""
    pipelines = build_pipelines_type()
    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        out_path = f"{model_dir}/{MODEL_FILENAMES_TYPE[name]}"
        joblib.dump(pipe, out_path)
        print(f"  {name:25s} → {out_path}")
    return trained


# ── RUL (Remaining Useful Life) – régression ─────────────────────────────────

MODEL_FILENAMES_RUL = {
    'Régression Linéaire': 'linear_regression_rul.joblib',
    'Random Forest':       'random_forest_rul.joblib',
    'XGBoost':             'xgboost_rul.joblib',
    'MLP':                 'mlp_rul.joblib',
    'SVM':                 'svm_rul.joblib',
}


def build_pipelines_rul() -> dict:
    """Retourne les pipelines de régression pour estimer le RUL (heures restantes)."""
    return {
        'Régression Linéaire': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1.0)),
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('reg', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('reg', XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=42, n_jobs=-1, eval_metric='rmse',
            )),
        ]),
        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu',
                max_iter=500, random_state=42, early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', SVR(kernel='rbf', C=1.0))
        ]),
    }


def train_and_save_rul(X_train, y_train, model_dir: str = 'models') -> dict:
    """Entraîne les 3 régresseurs RUL et les sauvegarde."""
    pipelines = build_pipelines_rul()
    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        out_path = f"{model_dir}/{MODEL_FILENAMES_RUL[name]}"
        joblib.dump(pipe, out_path)
        print(f"  {name:25s} → {out_path}")
    return trained
