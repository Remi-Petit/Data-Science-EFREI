"""
Pipeline d'entraînement – exécutable en ligne de commande.

Usage (depuis IA/Sujet_2/) :
    python train_pipeline.py
"""
import os
import sys
import json

# Résolution des imports locaux quand on exécute le script directement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, engineer_features, get_train_test_split, FEATURES
from src.train import build_pipelines, cross_validate_models, train_and_save, MODEL_FILENAMES
from src.evaluate import (
    evaluate_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance,
    plot_threshold_analysis,
)


def main():
    # ── 1. Données ────────────────────────────────────────────────────────────
    print("=== Chargement et préparation des données ===")
    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"Train : {X_train.shape[0]} lignes  |  Test : {X_test.shape[0]} lignes")
    print(f"Taux de churn (train) : {y_train.mean():.2%}")
    print(f"Déséquilibre classes  : {(y_train == 0).sum()} No-Churn vs {(y_train == 1).sum()} Churn")

    # ── 2. Cross-validation ───────────────────────────────────────────────────
    print("\n=== Cross-validation (StratifiedKFold 5 folds) ===")
    models = build_pipelines()
    cv_results = cross_validate_models(models, X_train, y_train)
    for name, scores in cv_results.items():
        print(f"  {name:25s} | ROC-AUC CV = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── 3. Entraînement final + sauvegarde ────────────────────────────────────
    print("\n=== Entraînement final + sauvegarde des modèles ===")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    trained = train_and_save(models, X_train, y_train, model_dir=models_dir)

    # ── 4. Évaluation ─────────────────────────────────────────────────────────
    print("\n=== Évaluation sur le jeu de test ===")
    results_df = evaluate_models(trained, X_test, y_test)
    print(results_df.to_string())
    # ── 4b. Sauvegarde des stats par modèle ──────────────────────────────────
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_stats')
    os.makedirs(stats_dir, exist_ok=True)
    _name_map = {'Logistic Regression': 'logistic_regression', 'Random Forest': 'random_forest', 'XGBoost': 'xgboost', 'MLP': 'mlp'}
    for model_name, row in results_df.iterrows():
        key = _name_map.get(model_name, model_name.lower().replace(' ', '_'))
        with open(os.path.join(stats_dir, f'{key}.json'), 'w') as f:
            json.dump(row.to_dict(), f, indent=2)
    print(f"Stats sauvegardées dans : {stats_dir}")
    # ── 5. Visualisations (sauvegardées en PNG) ───────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig_cm = plot_confusion_matrices(trained, X_test, y_test)
    fig_cm.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=120)

    fig_roc = plot_roc_curves(trained, X_test, y_test)
    fig_roc.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=120)

    fig_pr = plot_pr_curves(trained, X_test, y_test)
    fig_pr.savefig(os.path.join(results_dir, 'pr_curves.png'), dpi=120)

    fi_figures = plot_feature_importance(trained, FEATURES)
    for model_name, fig in fi_figures.items():
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        fig.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=120)

    thr_figures = plot_threshold_analysis(trained, X_test, y_test)
    for model_name, fig in thr_figures.items():
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        fig.savefig(os.path.join(model_dir, 'threshold_analysis.png'), dpi=120)

    # Sauvegarde du tableau comparatif
    results_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'))

    print(f"\nFigures et métriques sauvegardées dans : {results_dir}")

    # ── 6. Génération automatique de models_config.json et labels.json ──────────────────
    _base = os.path.dirname(os.path.abspath(__file__))
    models_config = {
        'churn': {_name_map[k]: v for k, v in MODEL_FILENAMES.items()},
    }
    with open(os.path.join(_base, 'models_config.json'), 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2)
    labels = {v: k for k, v in _name_map.items()}
    with open(os.path.join(_base, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"models_config.json et labels.json mis à jour.")

    print("\nPipeline terminé ✓")


if __name__ == '__main__':
    main()
