import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np

INDEX_FILE = "index.jsonl"
FEATURE_FILE = "features.npy"
SHAP_FILE = "shap.npy"
META_FILE = "meta.jsonl"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_index(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def init_matrix_files(namespace_dir: str, feature_dim: int, shap_dim: int):
    """Create empty .npy matrices if they don't exist."""
    feat_path = os.path.join(namespace_dir, FEATURE_FILE)
    shap_path = os.path.join(namespace_dir, SHAP_FILE)
    if not os.path.exists(feat_path):
        np.save(feat_path, np.zeros((0, feature_dim), dtype="float32"))
    if not os.path.exists(shap_path):
        np.save(shap_path, np.zeros((0, shap_dim), dtype="float32"))


def append_case(namespace_dir: str, case_id: str, features: np.ndarray, shap_vec: np.ndarray, meta: Dict[str, Any]):
    """Append one case to the namespace store."""
    ensure_dir(namespace_dir)

    # grow matrices
    feat_path = os.path.join(namespace_dir, FEATURE_FILE)
    shap_path = os.path.join(namespace_dir, SHAP_FILE)
    feats = np.load(feat_path)
    shaps = np.load(shap_path)
    feats = np.vstack([feats, features.reshape(1, -1).astype("float32")])
    shaps = np.vstack([shaps, shap_vec.reshape(1, -1).astype("float32")])
    np.save(feat_path, feats)
    np.save(shap_path, shaps)

    # index & meta
    idx_path = os.path.join(namespace_dir, INDEX_FILE)
    append_jsonl(idx_path, {"case_id": case_id, "row": feats.shape[0] - 1})
    meta_path = os.path.join(namespace_dir, META_FILE)
    append_jsonl(meta_path, {case_id: meta})


def load_matrices(namespace_dir: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Load all stored matrices and metadata for retrieval.
    Returns:
        X (np.ndarray)      : Features matrix
        SHAP (np.ndarray)   : SHAP values matrix
        metas (list[dict])  : Metadata entries
        case_ids (list[str]): Case IDs
    """
    # Load features & shap
    feat_path = os.path.join(namespace_dir, FEATURE_FILE)
    shap_path = os.path.join(namespace_dir, SHAP_FILE)
    X = np.load(feat_path)
    SHAP = np.load(shap_path)

    # Load metadata
    metas = []
    meta_path = os.path.join(namespace_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metas.append(json.loads(line))

    # Load case IDs
    case_ids = []
    idx_path = os.path.join(namespace_dir, INDEX_FILE)
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    case_ids.append(entry.get("case_id"))

    return X, SHAP, metas, case_ids
