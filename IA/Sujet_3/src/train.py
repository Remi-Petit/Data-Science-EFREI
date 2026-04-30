"""
Pipelines ML pour la régression (ventes) et la classification (performance).
Modèles régression    : Linear Regression, Random Forest, XGBoost, MLP
Modèles classification: Random Forest, XGBoost (bonus)
"""
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor, XGBClassifier

MODEL_FILENAMES_REG = {
    'Linear Regression': 'linear_regression_sales.joblib',
    'Random Forest':     'random_forest_sales.joblib',
    'XGBoost':           'xgboost_sales.joblib',
    'MLP':               'mlp_sales.joblib',
}

MODEL_FILENAMES_CLS = {
    'Random Forest': 'random_forest_performance.joblib',
    'XGBoost':       'xgboost_performance.joblib',
}


def build_regression_pipelines() -> dict:
    """
    Retourne un dict {nom: Pipeline sklearn} pour la régression des ventes.
    Tous incluent un SimpleImputer pour robustesse.
    LR et MLP incluent un StandardScaler.
    """
    return {
        'Linear Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('reg', LinearRegression()),
        ]),

        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('reg', RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('reg', XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),

        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('reg', MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )),
        ]),
    }


def build_classification_pipelines() -> dict:
    """Pipelines de classification de performance campagne (Low/Medium/High)."""
    return {
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mlogloss',
            )),
        ]),
    }


def cross_validate_regression(
    pipelines: dict,
    X_train,
    y_train,
    cv: int = 5,
) -> dict:
    """Cross-validation 5-fold pour chaque pipeline de régression."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {}
    for name, pipe in pipelines.items():
        r2_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
        neg_rmse  = cross_val_score(
            pipe, X_train, y_train, cv=kf,
            scoring='neg_root_mean_squared_error', n_jobs=-1,
        )
        scores[name] = {
            'r2_mean':   round(float(np.mean(r2_scores)), 4),
            'r2_std':    round(float(np.std(r2_scores)),  4),
            'rmse_mean': round(float(-np.mean(neg_rmse)), 4),
            'rmse_std':  round(float(np.std(neg_rmse)),   4),
        }
    return scores


def train_and_save(
    pipelines: dict,
    X_train,
    y_train,
    filenames: dict,
    output_dir: str,
) -> dict:
    """Entraîne chaque pipeline et sauvegarde le modèle sérialisé."""
    os.makedirs(output_dir, exist_ok=True)
    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        path = os.path.join(output_dir, filenames[name])
        joblib.dump(pipe, path)
        trained[name] = pipe
    return trained
