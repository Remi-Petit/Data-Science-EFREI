"""
Pipelines ML, cross-validation et entraînement final.
Modèles : Logistic Regression, Random Forest, XGBoost, MLP (Deep Learning)
Stratégie déséquilibre : class_weight='balanced' + Stratified K-Fold
"""
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

MODEL_FILENAMES = {
    'Logistic Regression': 'logistic_regression_churn.joblib',
    'Random Forest':       'random_forest_churn.joblib',
    'XGBoost':             'xgboost_churn.joblib',
    'MLP':                 'mlp_churn.joblib',
}


def build_pipelines() -> dict:
    """
    Retourne un dict {nom: Pipeline sklearn}.
    Tous les pipelines incluent un imputer (médiane) pour robustesse.
    Les modèles sensibles à l'échelle (LR, MLP) intègrent un StandardScaler.
    class_weight='balanced' gère le déséquilibre de classes (~10 % churn).
    """
    return {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
            )),
        ]),

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
                scale_pos_weight=8,   # approx ratio majoritaire/minoritaire
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0,
            )),
        ]),

        'MLP': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )),
        ]),
    }


def cross_validate_models(
    models: dict,
    X_train,
    y_train,
    n_splits: int = 5,
) -> dict:
    """
    Lance une StratifiedKFold CV et retourne les scores ROC-AUC par modèle.
    StratifiedKFold préserve les proportions de la classe minoritaire.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1,
        )
        results[name] = scores
    return results


def train_and_save(
    models: dict,
    X_train,
    y_train,
    model_dir: str = 'models',
) -> dict:
    """Entraîne chaque pipeline, le sauvegarde en joblib et retourne les pipelines entraînés."""
    os.makedirs(model_dir, exist_ok=True)
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        out_path = os.path.join(model_dir, MODEL_FILENAMES[name])
        joblib.dump(pipe, out_path)
        print(f"  {name:25s} → {out_path}")
    return trained
