from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class Instance(BaseModel):
    # Ordered feature vector for your model
    features: List[float]
    feature_names: List[str]

class PredictionResult(BaseModel):
    y_pred: float
    proba: Optional[float] = None

class Explanation(BaseModel):
    shap_values: List[float]          # reasoning vector
    base_value: float
    topk: List[Dict[str, Any]]

class RetrievalConfig(BaseModel):
    alpha: float = 0.7                # weight for SHAP cosine vs feature cosine
    k: int = 5
    use_retrieval: bool = True
    namespace: str = "global_default"

class RetrievedCase(BaseModel):
    case_id: str
    similarity: float
    y_pred: Optional[float] = None
    shap_values: Optional[List[float]] = None
    features: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None

class ExplainResponse(BaseModel):
    prediction: PredictionResult
    explanation: Explanation
    similar_cases: Optional[List[RetrievedCase]] = None
    ood_flag: bool = False
    ood_reason: Optional[str] = None
