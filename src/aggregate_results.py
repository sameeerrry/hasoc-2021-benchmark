"""
Walk results/ directory, aggregate metrics across seeds, produce a paper-ready table.

Usage:
    python src/aggregate_results.py --results_dir results --out results_summary.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results_summary.csv")
    args = ap.parse_args()

    rows = []
    for metrics_file in Path(args.results_dir).rglob("metrics.json"):
        try:
            with open(metrics_file) as f:
                m = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed {metrics_file}: {e}")
            continue

        run_name = metrics_file.parent.name
        # Directory format: {model}_seed{seed}
        if "_seed" in run_name:
            model_part, _, seed_part = run_name.rpartition("_seed")
        else:
            model_part, seed_part = run_name, "NA"

        rows.append({
            "model": m.get("model", model_part),
            "seed": int(seed_part) if seed_part.isdigit() else seed_part,
            "macro_f1": m.get("test_macro_f1"),
            "macro_precision": m.get("test_macro_precision"),
            "macro_recall": m.get("test_macro_recall"),
            "binary_f1": m.get("test_binary_f1"),
            "weighted_f1": m.get("test_weighted_f1"),
            "accuracy": m.get("test_accuracy"),
            "n_params_M": m.get("n_params_M"),
            "train_time_sec": m.get("train_time_sec"),
        })

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out.replace(".csv", "_raw.csv"), index=False)

    # Aggregate: mean ± std across seeds per model
    metric_cols = ["macro_f1", "macro_precision", "macro_recall",
                   "binary_f1", "weighted_f1", "accuracy"]
    agg = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)

    # Flatten
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = df.groupby("model").size()
    agg = agg.reset_index().sort_values("macro_f1_mean", ascending=False)
    agg.to_csv(args.out, index=False)

    print("\n=== Results summary (mean ± std across seeds) ===")
    for _, r in agg.iterrows():
        print(
            f"  {r['model']:<50s}  "
            f"macro-F1 = {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}  "
            f"(n={r['n_seeds']})"
        )

    print(f"\nRaw per-seed: {args.out.replace('.csv', '_raw.csv')}")
    print(f"Aggregated: {args.out}")

    # LaTeX table snippet for the paper
    latex = "\\begin{tabular}{lccc}\n\\toprule\nModel & Macro-F1 & Binary-F1 & Accuracy \\\\\n\\midrule\n"
    for _, r in agg.iterrows():
        latex += (f"{r['model']} & "
                  f"{r['macro_f1_mean']:.4f} $\\pm$ {r['macro_f1_std']:.4f} & "
                  f"{r['binary_f1_mean']:.4f} $\\pm$ {r['binary_f1_std']:.4f} & "
                  f"{r['accuracy_mean']:.4f} $\\pm$ {r['accuracy_std']:.4f} \\\\\n")
    latex += "\\bottomrule\n\\end{tabular}\n"

    with open(args.out.replace(".csv", "_latex.tex"), "w") as f:
        f.write(latex)
    print(f"LaTeX table: {args.out.replace('.csv', '_latex.tex')}")


if __name__ == "__main__":
    main()
