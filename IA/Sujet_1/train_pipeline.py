"""
Pipeline d'entraînement – exécutable en ligne de commande.

Usage (depuis IA/Sujet_1/) :
    python train_pipeline.py
"""
import os
import sys
import json

# Résolution des imports locaux quand on exécute le script directement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, engineer_features, get_train_test_split, get_type_train_test_split, get_rul_train_test_split, FEATURES
from src.train import build_pipelines, cross_validate_models, train_and_save, train_and_save_type, train_and_save_rul, MODEL_FILENAMES, MODEL_FILENAMES_TYPE, MODEL_FILENAMES_RUL
from src.evaluate import evaluate_models, evaluate_models_type, evaluate_models_rul, plot_confusion_matrices, plot_roc_curves, plot_feature_importance


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
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    trained = train_and_save(models, X_train, y_train, model_dir=models_dir)

    # ── 3b. Modèles failure_type (cause de panne, un par algo) ───────────────
    print("\n=== Entraînement des modèles failure_type ===")
    X_train_t, X_test_t, y_train_t, y_test_t = get_type_train_test_split(df)
    trained_type = train_and_save_type(X_train_t, y_train_t, model_dir=models_dir)

    # ── 3c. Modèles RUL (Remaining Useful Life – régression) ─────────────────
    print("\n=== Entraînement des modèles RUL ===")
    X_train_r, X_test_r, y_train_r, y_test_r = get_rul_train_test_split(df)
    trained_rul = train_and_save_rul(X_train_r, y_train_r, model_dir=models_dir)

    # ── 4. Évaluation failure_24h ────────────────────────────────────────────
    print("\n=== Évaluation failure_24h sur le jeu de test ===")
    results_df = evaluate_models(trained, X_test, y_test)
    print(results_df.to_string())

    # Sauvegarde stats failure_24h
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_stats', 'failure_24h')
    os.makedirs(stats_dir, exist_ok=True)
    _name_map = {'Logistic Regression': 'logistic_regression', 'Random Forest': 'random_forest', 'XGBoost': 'xgboost'}
    for model_name, row in results_df.iterrows():
        key = _name_map.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(stats_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)
    print(f"Stats failure_24h sauvegardées dans : {stats_dir}")

    # ── 4b. Évaluation failure_type ─────────────────────────────────────────
    print("\n=== Évaluation failure_type sur le jeu de test ===")
    results_type_df = evaluate_models_type(trained_type, X_test_t, y_test_t)
    print(results_type_df.to_string())

    # Sauvegarde stats failure_type
    stats_type_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_stats', 'failure_type')
    os.makedirs(stats_type_dir, exist_ok=True)
    for model_name, row in results_type_df.iterrows():
        key = _name_map.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(stats_type_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)
    print(f"Stats failure_type sauvegardées dans : {stats_type_dir}")

    # ── 4c. Évaluation RUL ───────────────────────────────────────────────────
    print("\n=== Évaluation RUL sur le jeu de test ===")
    results_rul_df = evaluate_models_rul(trained_rul, X_test_r, y_test_r)
    print(results_rul_df.to_string())

    stats_rul_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_stats', 'rul')
    os.makedirs(stats_rul_dir, exist_ok=True)
    _rul_name_map = {'Régression Linéaire': 'logistic_regression', 'Random Forest': 'random_forest', 'XGBoost': 'xgboost'}
    for model_name, row in results_rul_df.iterrows():
        key = _rul_name_map.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(stats_rul_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)
    print(f"Stats RUL sauvegardées dans : {stats_rul_dir}")

    # ── 5. Visualisations (sauvegardées en PNG) ───────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig_cm = plot_confusion_matrices(trained, X_test, y_test)
    fig_cm.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=120)

    fig_roc = plot_roc_curves(trained, X_test, y_test)
    fig_roc.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=120)

    fi_figures = plot_feature_importance(trained, FEATURES)
    for model_name, fig in fi_figures.items():
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        fig.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=120)

    print(f"\nFigures sauvegardées dans : {results_dir}")

    # ── 6. Génération automatique de models_config.json ──────────────────────
    _base = os.path.dirname(os.path.abspath(__file__))
    models_config = {
        'failure_24h':  {_name_map[k]: {"file": v, "label": k} for k, v in MODEL_FILENAMES.items()},
        'failure_type': {_name_map[k]: {"file": v, "label": k} for k, v in MODEL_FILENAMES_TYPE.items()},
        'rul':          {_rul_name_map[k]: {"file": v, "label": k} for k, v in MODEL_FILENAMES_RUL.items()},
    }
    with open(os.path.join(_base, 'models_config.json'), 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2, ensure_ascii=False)
    print(f"models_config.json mis à jour.")

    print("\nPipeline terminé ✓")


if __name__ == '__main__':
    main()
