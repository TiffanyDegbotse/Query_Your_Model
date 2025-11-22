"""
Add user-scoped reasoning vectors to their own namespace.

Usage:
  python scripts/add_user_model.py \
    --model_path ~/user_model.pkl \
    --csv ~/user_data.csv \
    --features f1,f2,f3 \
    --namespace data/user_indices/user_123_my_model \
    --limit 500
"""
import argparse
import pandas as pd
import numpy as np
from core.model_loader import load_model, predict
from core.explain import explain_instance
from core.storage import ensure_dir, init_matrix_files, append_case
from core.utils import to_numpy, case_id_from_vector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    feat_names = args.features.split(",")
    df = pd.read_csv(args.csv)
    if args.limit < len(df):
        df = df.sample(args.limit, random_state=1337)

    X = df[feat_names].values.astype("float32")
    model = load_model(args.model_path)

    ensure_dir(args.namespace)
    init_matrix_files(args.namespace, feature_dim=X.shape[1], shap_dim=X.shape[1])

    bg = df[feat_names].sample(min(100, len(df)), random_state=0).values.astype("float32")

    for _, row in df.iterrows():
        x = row[feat_names].values.astype("float32")
        y_pred, _ = predict(model, x.reshape(1, -1))
        exp = explain_instance(model, x, feat_names, background_X=bg, top_k=8)
        cid = case_id_from_vector(x, prefix="usercase")
        append_case(args.namespace, cid, x, np.array(exp["shap_values"], dtype="float32"), {"y_pred": float(y_pred[0])})

    print(f"User namespace populated at: {args.namespace}")

if __name__ == "__main__":
    main()
