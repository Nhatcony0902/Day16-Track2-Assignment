"""LightGBM benchmark on the Kaggle Credit Card Fraud Detection dataset (Lab 16).

Usage (on the compute node):
    python3 benchmark.py                                  # default: ~/ml-benchmark/creditcard.csv
    python3 benchmark.py --data /path/to/creditcard.csv --out benchmark_result.json
"""
import argparse
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42

# LightGBM >= 4.7 deprecates eval_set in favour of eval_X/eval_y; keep eval_set for older versions
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM fraud-detection benchmark")
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the fraud class")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_time = time.perf_counter() - t0

    X = df.drop(columns=["Class"])
    y = df["Class"]
    print(f"Loaded {len(df):,} rows x {X.shape[1]} features in {load_time:.3f}s "
          f"(fraud rate {y.mean() * 100:.3f}%)")

    # 2. Split: 70% train / 10% validation (early stopping) / 20% test, stratified on the rare fraud class
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=SEED)

    # 3. Train
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        # With only ~0.17% fraud, near-empty-hessian leaves get extreme values and validation AUC
        # collapses after the first tree (best_iteration=1). Regularising leaves fixes it.
        min_child_weight=5,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    train_time = time.perf_counter() - t0
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    print(f"Training done in {train_time:.3f}s, best iteration = {best_iteration}")

    # 4. Evaluate on the held-out test set
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= args.threshold).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
    }

    # 5. Inference speed
    one_row = X_test.iloc[[0]]
    for _ in range(10):  # warm-up
        model.predict_proba(one_row)
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        latencies.append(time.perf_counter() - t0)
    latency_ms = float(np.median(latencies) * 1000)

    batch = X_test.iloc[:1000]
    batch_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict_proba(batch)
        batch_times.append(time.perf_counter() - t0)
    batch_ms = float(np.median(batch_times) * 1000)
    throughput = len(batch) / (batch_ms / 1000)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "lightgbm": lgb.__version__,
        },
        "dataset": {
            "path": args.data,
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "fraud_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "data_load_time_s": round(load_time, 4),
        "training_time_s": round(train_time, 4),
        "best_iteration": best_iteration,
        "threshold": args.threshold,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "inference": {
            "latency_1_row_ms": round(latency_ms, 4),
            "batch_1000_rows_ms": round(batch_ms, 4),
            "throughput_rows_per_s": round(throughput, 1),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    rows = [
        ("Thoi gian load data", f"{load_time:.3f} s"),
        ("Thoi gian training", f"{train_time:.3f} s"),
        ("Best iteration", str(best_iteration)),
        ("AUC-ROC", f"{metrics['auc_roc']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("F1-Score", f"{metrics['f1_score']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("Inference latency (1 row)", f"{latency_ms:.3f} ms"),
        ("Inference throughput (1000 rows)", f"{batch_ms:.3f} ms  (~{throughput:,.0f} rows/s)"),
    ]
    width = max(len(name) for name, _ in rows)
    print("\n" + "=" * (width + 30))
    print("LightGBM CPU Benchmark - Credit Card Fraud Detection")
    print("=" * (width + 30))
    for name, value in rows:
        print(f"{name:<{width}} | {value}")
    print("=" * (width + 30))
    print(f"Saved results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
