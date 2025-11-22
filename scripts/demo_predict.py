"""
Quick script to run a single prediction + explanation without the UI.

Usage:
  python scripts/demo_predict.py \
      --model_path path/to/model.pkl \
      --features 0.2,1.0,3.4,5.5 \
      --feat_names f1,f2,f3,f4 \
      --bg_csv path/to/bg.csv
"""
import argparse
import numpy as np
import pandas as pd
from core.model_loader import load_model, predict
from core.explain import explain_instance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--feat_names", required=True)
    ap.add_argument("--bg_csv", required=True)
    args = ap.parse_args()

    model = load_model(args.model_path)
    feat_names = args.feat_names.split(",")
    x = np.array([float(v) for v in args.features.split(",")], dtype="float32")
    bg = pd.read_csv(args.bg_csv)[feat_names].sample(100, replace=True, random_state=42).values.astype("float32")

    y_pred, proba = predict(model, x.reshape(1, -1))
    exp = explain_instance(model, x, feat_names, background_X=bg, top_k=8)
    print("Prediction:", float(y_pred[0]))
    if proba is not None:
        print("Probabilities:", proba)
    print("Base value:", exp["base_value"])
    print("Top contributions:")
    for t in exp["topk"]:
        print(t)

if __name__ == "__main__":
    main()
