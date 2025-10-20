from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import numpy as np

from Query_Your_Model.core.schemas import RetrievalConfig, ExplainResponse
from Query_Your_Model.core.model_loader import load_model
from Query_Your_Model.core.explain import explain_instance
from Query_Your_Model.core.retrieval import retrieve_topk
from Query_Your_Model.core.utils import safe_proba_to_scalar

app = FastAPI(title="Reasoning-RAG XAI API")

# Cached globals
MODEL = None
FEATURE_NAMES: Optional[List[str]] = None
BACKGROUND = None
NAMESPACE = "Query_Your_Model/data/base_indices/iris_global"

# --- Target name mappings (extend per dataset/model) ---
TARGET_NAMES = {
    "iris": ["setosa", "versicolor", "virginica"],
    # add more datasets here if needed
}


class ExplainRequest(BaseModel):
    model_path: str
    feature_names: List[str]
    features: List[float]
    namespace: Optional[str] = None
    retrieval: Optional[RetrievalConfig] = None
    background_path: Optional[str] = None


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    global MODEL, FEATURE_NAMES, BACKGROUND

    # Load model if not cached
    if (MODEL is None) or (FEATURE_NAMES != req.feature_names):
        MODEL = load_model(req.model_path)
        FEATURE_NAMES = req.feature_names
        BACKGROUND = None  # optionally load background data

    # Convert input features
    x = np.asarray(req.features, dtype="float32").reshape(1, -1)

    # Prediction & probability
    y_class = 0
    proba_scalar = None
    try:
        y_pred = MODEL.predict(x)
        y_class = int(y_pred[0])

        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(x)
            proba_scalar = float(proba[0][y_class])
    except Exception as e:
        print("Prediction error:", e)

    # --- Map class ID -> human-readable label ---
    model_key = "iris" if "iris" in req.model_path.lower() else None
    if model_key and model_key in TARGET_NAMES:
        y_label = TARGET_NAMES[model_key][y_class]
    else:
        y_label = str(y_class)

    # SHAP explanation
    exp = explain_instance(
        MODEL,
        x[0],
        FEATURE_NAMES,
        background_X=(BACKGROUND if BACKGROUND is not None else x),
    )

    # Retrieval
    similar = None
    ns = req.namespace or NAMESPACE
    if req.retrieval and req.retrieval.use_retrieval:
        shap_q = np.array(exp["shap_values"], dtype="float32")
        similar = retrieve_topk(ns, shap_q, x[0], alpha=req.retrieval.alpha, k=req.retrieval.k)

        # also map labels for retrieved cases
        if model_key and model_key in TARGET_NAMES:
            for case in similar:
                if case.get("y_pred") is not None:
                    try:
                        case["y_pred"] = TARGET_NAMES[model_key][int(case["y_pred"])]
                    except Exception:
                        case["y_pred"] = str(case["y_pred"])

    return ExplainResponse(
        prediction={
            "y_pred": y_label,     # now returns "setosa", "versicolor", etc.
            "proba": proba_scalar,
        },
        explanation=exp,
        similar_cases=similar or [],
        ood_flag=False
    )
