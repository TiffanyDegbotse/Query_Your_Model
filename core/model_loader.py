import joblib
from typing import Any, Tuple, Optional
import numpy as np

def load_model(path: str) -> Any:
    """Load a pickled sklearn-compatible model."""
    model = joblib.load(path)
    return model

def predict(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (pred, proba_or_none). Handles regressors & classifiers."""
    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    return y_pred, proba
