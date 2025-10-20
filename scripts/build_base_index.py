"""
Precompute a 'global' reasoning space from a baseline model + dataset.

Usage:
  python scripts/build_base_index.py \
      --model_path path/to/model.pkl \
      --csv path/to/data.csv \
      --features col1,col2,col3 \
      --target target_col \
      --namespace data/base_indices/recidivism_global \
      --sample 2000
"""
# Query_Your_Model/scripts/build_base_index.py
import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import pandas as pd
import numpy as np
from ..core.model_loader import load_model, predict
from ..core.explain import explain_instance
from ..core.storage import ensure_dir, init_matrix_files, append_case
from ..core.utils import case_id_from_vector


# Hardcoded defaults for Iris demo
MODEL_PATH = "Query_Your_Model/model_data/model.pkl"
CSV_PATH = "Query_Your_Model/model_data/data.csv"
FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
TARGET = "target"
NAMESPACE = "Query_Your_Model/data/base_indices/iris_global"
SAMPLE = 100   # how many rows to sample

def main():
    print("Building reasoning index...")
    df = pd.read_csv(CSV_PATH)
    if SAMPLE and SAMPLE < len(df):
        df = df.sample(SAMPLE, random_state=42)

    X = df[FEATURES].values
    model = load_model(MODEL_PATH)

    ensure_dir(NAMESPACE)
    init_matrix_files(NAMESPACE, feature_dim=len(FEATURES), shap_dim=len(FEATURES))


    bg = df[FEATURES].sample(min(100, len(df)), random_state=0).values.astype("float32")

    for i, row in df.iterrows():
        x = row[FEATURES].values.astype("float32")
        y_pred, _ = predict(model, x.reshape(1, -1))
        exp = explain_instance(model, x, FEATURES, background_X=bg, top_k=8)
        shap_vec = np.array(exp["shap_values"], dtype="float32")
        cid = case_id_from_vector(x, prefix="iris")
        meta = {"y_pred": float(y_pred[0])}
        append_case(NAMESPACE, cid, x, shap_vec, meta)

    print(f"Done! Index saved to {NAMESPACE}")

if __name__ == "__main__":
    main()
