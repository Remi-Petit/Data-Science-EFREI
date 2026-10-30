"""
Convertit les modèles .joblib en .onnx (consommés par le backend Rust) et écrit
la liste des classes (<fichier>.classes.json) pour les classifieurs.

Usage (après l'entraînement) :
    python IA/export_onnx.py          # tous les sujets
    python IA/export_onnx.py 1 3      # sujets choisis
"""
import copy
import json
import os
import sys

import joblib
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from xgboost import XGBClassifier, XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'Sujet_1'))  # classe XGBClassifierWithEncoder (pickle)

update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']},
)
update_registered_converter(
    XGBRegressor, 'XGBoostXGBRegressor',
    calculate_linear_regressor_output_shapes, convert_xgboost,
)

# sujet -> (dossier, [groupes de models_config.json])
SUJETS = {
    1: ('Sujet_1', ['failure_24h', 'failure_type', 'rul']),
    2: ('Sujet_2', ['churn']),
    3: ('Sujet_3', ['regression']),  # le groupe 'classification' n'est pas servi par l'API
}


def _unwrap_xgb(pipe):
    """Remplace XGBClassifierWithEncoder par son XGBClassifier interne ; retourne les classes."""
    final = pipe.steps[-1][1]
    if hasattr(final, 'xgb_'):
        classes = [str(c) for c in final.classes_]
        pipe.steps[-1] = (pipe.steps[-1][0], final.xgb_)
        return classes
    return None


def _prepare_xgb(final):
    if isinstance(final, (XGBClassifier, XGBRegressor)):
        final.get_booster().feature_names = None  # onnxmltools exige f0, f1, ...


def export(joblib_path: str) -> None:
    pipe = copy.deepcopy(joblib.load(joblib_path))
    classes = _unwrap_xgb(pipe)
    final = pipe.steps[-1][1]
    _prepare_xgb(final)
    is_clf = hasattr(final, 'predict_proba')
    if is_clf and classes is None:
        classes = [str(c) for c in final.classes_]

    n_features = pipe.steps[0][1].n_features_in_
    options = {id(final): {'zipmap': False}} if is_clf else None
    onx = convert_sklearn(
        pipe, initial_types=[('X', FloatTensorType([None, n_features]))],
        options=options, target_opset={'': 17, 'ai.onnx.ml': 3},
    )
    base = joblib_path[:-len('.joblib')]
    with open(base + '.onnx', 'wb') as f:
        f.write(onx.SerializeToString())
    if is_clf:
        with open(base + '.classes.json', 'w', encoding='utf-8') as f:
            json.dump(classes, f)
    print(f"  {os.path.basename(joblib_path)} -> .onnx")


def main():
    selected = [int(a) for a in sys.argv[1:]] or list(SUJETS)
    for sid in selected:
        folder, groups = SUJETS[sid]
        base = os.path.join(HERE, folder)
        with open(os.path.join(base, 'models_config.json'), encoding='utf-8') as f:
            config = json.load(f)
        print(f"== {folder}")
        for group in groups:
            for entry in config.get(group, {}).values():
                filename = entry['file'] if isinstance(entry, dict) else entry
                path = os.path.join(base, 'models', filename)
                if os.path.isfile(path):
                    export(path)
                else:
                    print(f"  absent, ignoré : {path}")


if __name__ == '__main__':
    main()
