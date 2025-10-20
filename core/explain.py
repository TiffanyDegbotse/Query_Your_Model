from typing import List, Dict, Any
import numpy as np
import shap


def _pick_explainer(model, X_background: np.ndarray):
    """
    Choose an appropriate SHAP explainer.
    - TreeExplainer for tree-based models
    - LinearExplainer for linear models
    - KernelExplainer fallback (slow but general)
    """
    try:
        import xgboost  # noqa: F401
        is_tree = hasattr(model, "get_booster") or "xgb" in type(model).__name__.lower()
    except Exception:
        is_tree = False

    is_tree = is_tree or any(
        s in type(model).__name__.lower()
        for s in ["randomforest", "gradientboost", "gbm", "lightgbm", "catboost"]
    )

    if is_tree:
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    is_linear = "linear" in type(model).__name__.lower() or hasattr(model, "coef_")
    if is_linear:
        return shap.LinearExplainer(model, X_background)

    # Fallback for anything else
    return shap.KernelExplainer(model.predict, X_background)


def explain_instance(
    model,
    x: np.ndarray,
    feature_names: List[str],
    background_X: np.ndarray,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Compute SHAP for a single instance x (shape: (n_features,)).
    Always reduces SHAP output to a vector of length = n_features.
    Handles multiclass by averaging across classes.
    """
    x = x.reshape(1, -1)
    explainer = _pick_explainer(model, background_X)

    values = explainer.shap_values(x)

    # SHAP returns different shapes depending on model type
    if isinstance(values, list):  # multiclass -> list of arrays
        # stack into shape (n_classes, n_samples, n_features)
        values_arr = np.stack(values, axis=0)
        # average across classes -> shape (n_samples, n_features)
        values_arr = np.mean(values_arr, axis=0)
    else:
        values_arr = values  # already (n_samples, n_features)

    # Always flatten to 1D vector
    shap_vec = np.array(values_arr[0]).reshape(-1)

    # Ensure length matches feature_names
    n_features = len(feature_names)
    if len(shap_vec) != n_features:
        shap_vec = shap_vec[:n_features]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.mean(base_value))

    # Top-k by absolute impact
    abs_imp = np.abs(shap_vec)
    idx = np.argsort(-abs_imp)[:top_k].ravel()

    top = []
    for i in idx:
        i = int(i)
        if i >= n_features:  # safety check
            continue

        shap_val = shap_vec[i]
        if isinstance(shap_val, (np.ndarray, list)):
            shap_val = float(np.mean(shap_val))
        else:
            shap_val = float(shap_val)

        abs_val = abs_imp[i]
        if isinstance(abs_val, (np.ndarray, list)):
            abs_val = float(np.mean(abs_val))
        else:
            abs_val = float(abs_val)

        top.append({
            "feature": feature_names[i],
            "value": float(x[0, i]),
            "shap": shap_val,
            "abs_impact": abs_val,
        })

    return {
        "shap_values": shap_vec.tolist(),
        "base_value": float(base_value),
        "topk": top,
    }
