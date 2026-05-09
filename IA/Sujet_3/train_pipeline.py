"""
Script principal d'entraînement – Sujet 3 : Optimisation du ROI Marketing.

Tâche principale : Régression – prédiction des ventes (Sales)
Tâche bonus      : Classification – performance campagne (Low / Medium / High)

Exécution :
    python train_pipeline.py
"""
import os
import sys

_src_dir = os.path.dirname(os.path.abspath(__file__))
import json
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.preprocessing import load_data, engineer_features, get_train_test_split, FEATURES
from src.train import (
    build_regression_pipelines,
    build_classification_pipelines,
    cross_validate_regression,
    train_and_save,
    MODEL_FILENAMES_REG,
    MODEL_FILENAMES_CLS,
)
from src.evaluate import (
    evaluate_regression,
    evaluate_classification,
    get_feature_importance,
)

MODELS_DIR  = os.path.join(_src_dir, 'models')
RESULTS_DIR = os.path.join(_src_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 55)
    print("  Sujet 3 – Marketing ROI Optimisation")
    print("=" * 55)

    # ── 1. Chargement et feature engineering ──────────────────────────────────
    print("\n[1/4] Chargement et feature engineering...")
    df_raw = load_data()
    df     = engineer_features(df_raw)
    print(f"  Dataset nettoyé : {len(df)} lignes")
    print(f"  Features        : {FEATURES}")

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = \
        get_train_test_split(df)
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # ── 2. Régression – prédiction des ventes ─────────────────────────────────
    print("\n[2/4] Régression – prédiction des ventes...")
    reg_pipes = build_regression_pipelines()

    print("  Cross-validation 5-fold...")
    cv_scores = cross_validate_regression(reg_pipes, X_train, y_reg_train)
    for name, s in cv_scores.items():
        print(f"    {name:<22} R²={s['r2_mean']:.4f} ±{s['r2_std']:.4f}  "
              f"RMSE={s['rmse_mean']:.2f} ±{s['rmse_std']:.2f}")

    print("  Entraînement final et sauvegarde des modèles...")
    trained_reg = train_and_save(reg_pipes, X_train, y_reg_train, MODEL_FILENAMES_REG, MODELS_DIR)

    df_reg = evaluate_regression(trained_reg, X_test, y_reg_test)
    print("\n  Résultats sur le jeu de test :")
    print(df_reg.to_string())
    df_reg.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_regression.csv'))

    # Sauvegarde des stats régression par modèle
    _stats_reg_dir = os.path.join(_src_dir, 'models_stats', 'regression')
    os.makedirs(_stats_reg_dir, exist_ok=True)
    _name_map_reg = {'Linear Regression': 'linear_regression', 'Random Forest': 'random_forest', 'XGBoost': 'xgboost', 'MLP': 'mlp'}
    for model_name, row in df_reg.iterrows():
        key = _name_map_reg.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(_stats_reg_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)

    # Feature importance (Random Forest)
    fi = get_feature_importance(trained_reg['Random Forest'], FEATURES)
    if not fi.empty:
        rf_dir = os.path.join(RESULTS_DIR, 'random_forest')
        os.makedirs(rf_dir, exist_ok=True)
        fi.to_csv(os.path.join(rf_dir, 'feature_importance.csv'), index=False)
        print("\n  Feature importance – Random Forest :")
        print(fi.to_string(index=False))

    # ── 3. Classification – performance campagne (BONUS) ──────────────────────
    print("\n[3/4] Classification – performance campagne [BONUS]...")
    cls_pipes   = build_classification_pipelines()
    trained_cls = train_and_save(cls_pipes, X_train, y_cls_train, MODEL_FILENAMES_CLS, MODELS_DIR)

    df_cls = evaluate_classification(trained_cls, X_test, y_cls_test)
    print("\n  Résultats classification :")
    print(df_cls.to_string())
    df_cls.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_classification.csv'))

    # Sauvegarde des stats classification par modèle
    _stats_cls_dir = os.path.join(_src_dir, 'models_stats', 'classification')
    os.makedirs(_stats_cls_dir, exist_ok=True)
    _name_map_cls = {'Random Forest': 'random_forest', 'XGBoost': 'xgboost'}
    for model_name, row in df_cls.iterrows():
        key = _name_map_cls.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(_stats_cls_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)

    # ── 4. Fin ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Modèles sauvegardés → {MODELS_DIR}")
    print(f"      Résultats  sauvegardés → {RESULTS_DIR}")
    print("\n" + "=" * 55)
    print("  Pipeline terminé avec succès.")
    print("=" * 55)


if __name__ == '__main__':
    main()
