"""Fresh-process inspection of artifacts created by these checks only."""

import json
import sys
from pathlib import Path

import joblib
from arfs.gbm import GradientBoosting

model = GradientBoosting(show_learning_curve=False)
model.load(sys.argv[1])
X = joblib.load(sys.argv[2])
result = {
    "raw": model.predict_raw(X, raw_score=True).tolist(),
    "native": model.predict_raw(X).tolist(),
    "is_init_score": model.is_init_score,
    "cat_feat": model.cat_feat,
    "feature_names": model.model.feature_name(),
    "objective": model.model_params["objective"],
    "best_iteration": model.model.best_iteration,
}
try:
    result["prediction"] = model.predict(X).tolist()
except Exception as exc:
    result["prediction_error"] = type(exc).__name__
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
