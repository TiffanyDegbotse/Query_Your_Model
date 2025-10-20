from typing import Dict, Any, List
import numpy as np
from .storage import load_matrices


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def combined_similarity(
    shap_q: np.ndarray,
    feat_q: np.ndarray,
    shap_i: np.ndarray,
    feat_i: np.ndarray,
    alpha: float
) -> float:
    """similarity = alpha * cos(SHAP) + (1 - alpha) * cos(features)"""
    return alpha * _cosine(shap_q, shap_i) + (1.0 - alpha) * _cosine(feat_q, feat_i)


def retrieve_topk(
    namespace: str,
    shap_q: np.ndarray,
    x_q: np.ndarray,
    alpha: float = 0.5,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar cases from a namespace.
    Returns dicts with case_id, similarity, y_pred, shap_values, features, meta.
    """
    # Load stored matrices and metadata
    X, SHAP, metas, case_ids = load_matrices(namespace)

    # Flatten metas into a dict keyed by case_id
    meta_dict: Dict[str, Dict[str, Any]] = {}
    for m in metas:
        if isinstance(m, dict):
            meta_dict.update(m)

    sims: List[Dict[str, Any]] = []
    for i, cid in enumerate(case_ids):
        feat = X[i]
        shap = SHAP[i]

        # compute similarity
        score = combined_similarity(shap_q, x_q, shap, feat, alpha=alpha)

        # get meta (safe fallback)
        m = meta_dict.get(cid, {})

        sims.append({
            "case_id": cid,
            "similarity": float(score),
            "y_pred": m.get("y_pred"),
            "shap_values": shap.tolist(),
            "features": feat.tolist(),
            "meta": m
        })

    # sort and return top-k
    sims = sorted(sims, key=lambda d: -d["similarity"])
    return sims[:k]


def ood_score(shap_query: np.ndarray, shaps_matrix: np.ndarray) -> float:
    """Simple OOD heuristic: 1 - max cosine against corpus SHAPs."""
    if shaps_matrix.size == 0:
        return 1.0
    best = -1.0
    for i in range(shaps_matrix.shape[0]):
        c = _cosine(shap_query, shaps_matrix[i])
        if c > best:
            best = c
    return float(1.0 - best)
