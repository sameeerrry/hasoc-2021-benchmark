"""
TF-IDF + Linear SVM baseline for HASOC 2021 Subtask 1.

Usage:
    python src/train_tfidf.py --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, classification_report, confusion_matrix
)

from clean import clean_for_tfidf

LABEL2ID = {"NONE": 0, "HOF": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/processed/train.csv")
    ap.add_argument("--val_csv", default="data/processed/val.csv")
    ap.add_argument("--test_csv", default="data/processed/test.csv")
    ap.add_argument("--output_root", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifier", choices=["svm", "logreg"], default="svm")
    args = ap.parse_args()

    out_dir = Path(args.output_root) / f"tfidf_{args.classifier}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    # Additional aggressive cleaning for classical baseline
    for df in (train_df, val_df, test_df):
        df["text_clean"] = df["text"].astype(str).apply(clean_for_tfidf)
        df["label_id"] = df["label"].map(LABEL2ID)

    # Combine train + val for final fit (we don't early stop for classical)
    X_trainval = pd.concat([train_df["text_clean"], val_df["text_clean"]])
    y_trainval = pd.concat([train_df["label_id"], val_df["label_id"]])

    if args.classifier == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=args.seed,
                        max_iter=5000)
    else:
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 random_state=args.seed, max_iter=5000)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("clf", clf),
    ])

    pipe.fit(X_trainval, y_trainval)

    preds = pipe.predict(test_df["text_clean"])
    y_true = test_df["label_id"].values

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    metrics = {
        "test_macro_f1": float(f1),
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_binary_f1": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, preds)),
        "model": f"tfidf_{args.classifier}",
        "seed": args.seed,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(
            y_true, preds, target_names=["NONE", "HOF"], digits=4
        ))

    np.save(out_dir / "confusion_matrix.npy", confusion_matrix(y_true, preds))

    pd.DataFrame({
        "tweet_id": test_df["tweet_id"].values,
        "text": test_df["text"].values,
        "true": y_true,
        "pred": preds,
    }).to_csv(out_dir / "predictions.csv", index=False)

    print(f"\nTest macro-F1: {f1:.4f}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
