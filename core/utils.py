import hashlib
from typing import List
import numpy as np

def case_id_from_vector(x: np.ndarray, prefix: str = "case") -> str:
    h = hashlib.md5(x.tobytes()).hexdigest()[:10]
    return f"{prefix}_{h}"

def to_numpy(lst, dtype="float32"):
    return np.asarray(lst, dtype=dtype)

def safe_proba_to_scalar(proba, positive_index: int = 1):
    """Return a single probability for binary classifiers when possible."""
    if proba is None:
        return None
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return float(arr[0, positive_index])
    # fallback: average
    return float(arr.mean())
