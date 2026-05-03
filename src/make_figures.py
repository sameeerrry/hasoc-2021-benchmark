"""
Generate paper-ready figures from the results/ folder.

Outputs (saved to figures/):
    1. confusion_matrices_per_seed.png   - 1 confusion matrix per (model, seed)
    2. confusion_matrices_aggregated.png - 1 confusion matrix per model (summed across seeds)
    3. macro_f1_comparison.png           - bar chart, mean +/- std macro-F1 by model
    4. all_metrics_comparison.png        - grouped bar chart, multiple metrics by model
    5. per_class_f1.png                  - HOF vs NONE F1 per model

Usage:
    python src/make_figures.py
    python src/make_figures.py --results_dir results --out_dir figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Display order for the paper. Update this list if you add/remove models.
MODEL_ORDER = [
    "tfidf_svm",
    "tfidf_logreg",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "google/muril-base-cased",
    "ai4bharat/IndicBERTv2-MLM-only",
    "l3cube-pune/hing-roberta",
]

# Pretty names for axis labels and legends.
PRETTY = {
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_logreg": "TF-IDF + LogReg",
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R base",
    "google/muril-base-cased": "MuRIL",
    "ai4bharat/IndicBERTv2-MLM-only": "IndicBERT v2",
    "l3cube-pune/hing-roberta": "HingRoBERTa",
}


def configure_style():
    """Set sane plot defaults for paper figures."""
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_runs(results_dir):
    """Walk results/ and return list of dicts, one per run."""
    runs = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse {metrics_path}: {e}")
            continue

        # Directory name is e.g. "bert-base-multilingual-cased_seed42" or
        # "google_muril-base-cased_seed42".
        name = run_dir.name
        if "_seed" not in name:
            continue
        model_part, _, seed_part = name.rpartition("_seed")
        # Reverse the "/" -> "_" mangling for HF model names with org prefix
        # (e.g., "google_muril-base-cased" came from "google/muril-base-cased").
        model_name = m.get("model", model_part)

        cm_path = run_dir / "confusion_matrix.npy"
        cm = None
        if cm_path.exists():
            try:
                cm = np.load(cm_path)
            except Exception:
                cm = None

        runs.append({
            "model": model_name,
            "seed": int(seed_part) if seed_part.isdigit() else seed_part,
            "metrics": m,
            "cm": cm,
            "run_dir": run_dir,
        })
    return runs


def aggregate_by_model(runs):
    """Group runs by model; compute mean/std for each metric."""
    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)

    summary = {}
    for model, rs in by_model.items():
        # Pull every test_* metric across seeds
        keys = [k for k in rs[0]["metrics"] if k.startswith("test_")
                and isinstance(rs[0]["metrics"][k], (int, float))]
        agg = {}
        for k in keys:
            vals = [r["metrics"][k] for r in rs if k in r["metrics"]]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

        # Aggregate confusion matrices: sum across seeds (assumes same test set).
        cms = [r["cm"] for r in rs if r["cm"] is not None]
        agg_cm = np.sum(cms, axis=0) if cms else None

        summary[model] = {
            "n_seeds": len(rs),
            "metrics": agg,
            "cm_total": agg_cm,
            "runs": rs,
        }
    return summary


def get_models_in_order(summary):
    """Return models present, ordered by MODEL_ORDER, with unknowns appended."""
    present = list(summary.keys())
    ordered = [m for m in MODEL_ORDER if m in summary]
    extras = [m for m in present if m not in MODEL_ORDER]
    return ordered + sorted(extras)


# --------------------------------------------------------------------------
# Figure 1: per-seed confusion matrices grid (one row per model)
# --------------------------------------------------------------------------
def plot_confusion_matrices_per_seed(summary, out_path):
    models = get_models_in_order(summary)
    if not models:
        print("[WARN] No models found; skipping per-seed CM figure.")
        return

    n_seeds_max = max(s["n_seeds"] for s in summary.values())
    fig, axes = plt.subplots(
        len(models), n_seeds_max,
        figsize=(2.6 * n_seeds_max, 2.6 * len(models)),
        squeeze=False,
    )

    for row, model in enumerate(models):
        runs = sorted(summary[model]["runs"], key=lambda r: r["seed"])
        for col in range(n_seeds_max):
            ax = axes[row][col]
            if col >= len(runs):
                ax.axis("off")
                continue
            run = runs[col]
            cm = run["cm"]
            if cm is None:
                ax.text(0.5, 0.5, "no CM", ha="center", va="center",
                        transform=ax.transAxes)
                ax.axis("off")
                continue

            _draw_cm(ax, cm, classes=["NONE", "HOF"],
                     title=f"seed {run['seed']}")

            if col == 0:
                ax.set_ylabel(PRETTY.get(model, model), fontsize=11, weight="bold")

    fig.suptitle("Confusion matrices per (model, seed) on test set",
                 fontsize=13, weight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Figure 2: per-model aggregated confusion matrices (sum across seeds)
# --------------------------------------------------------------------------
def plot_confusion_matrices_aggregated(summary, out_path):
    models = get_models_in_order(summary)
    if not models:
        return

    n_cols = min(3, len(models))
    n_rows = (len(models) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i, model in enumerate(models):
        ax = axes[i // n_cols][i % n_cols]
        cm = summary[model]["cm_total"]
        n_seeds = summary[model]["n_seeds"]
        if cm is None:
            ax.axis("off")
            continue
        _draw_cm(ax, cm, classes=["NONE", "HOF"],
                 title=f"{PRETTY.get(model, model)}\n(summed over {n_seeds} seeds)")

    # Hide any unused axes
    for j in range(len(models), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def _draw_cm(ax, cm, classes, title=""):
    """Helper: draw a 2x2 confusion matrix on the given axis."""
    cm = np.asarray(cm)
    # Row-normalize for color, but show absolute counts as labels.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, color=color)


# --------------------------------------------------------------------------
# Figure 3: macro-F1 bar chart (mean +/- std)
# --------------------------------------------------------------------------
def plot_macro_f1(summary, out_path):
    models = get_models_in_order(summary)
    means, stds, labels = [], [], []
    for m in models:
        agg = summary[m]["metrics"].get("test_macro_f1")
        if not agg:
            continue
        means.append(agg["mean"])
        stds.append(agg["std"])
        labels.append(PRETTY.get(m, m))

    if not means:
        print("[WARN] No macro_f1 to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color="#4C72B0", edgecolor="black", linewidth=0.5,
                  error_kw={"linewidth": 1.2})

    # Highlight best
    best_idx = int(np.argmax(means))
    bars[best_idx].set_color("#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (test set)")
    ax.set_title("Macro-F1 on test set: mean \u00b1 std across 3 seeds",
                 fontsize=12)
    ax.set_ylim(0, max(0.85, max(means) + max(stds) + 0.08))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate bars with their values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.012,
                f"{mean:.3f}\n\u00b1{std:.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Figure 4: grouped bar chart of multiple metrics per model
# --------------------------------------------------------------------------
def plot_all_metrics(summary, out_path):
    models = get_models_in_order(summary)
    metric_keys = ["test_macro_f1", "test_binary_f1",
                   "test_weighted_f1", "test_accuracy"]
    metric_pretty = {"test_macro_f1": "Macro-F1",
                     "test_binary_f1": "Binary-F1 (HOF)",
                     "test_weighted_f1": "Weighted-F1",
                     "test_accuracy": "Accuracy"}

    n_metrics = len(metric_keys)
    n_models = len(models)
    if n_models == 0:
        return

    width = 0.8 / n_metrics
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.4), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    for i, key in enumerate(metric_keys):
        means = [summary[m]["metrics"].get(key, {}).get("mean", 0) for m in models]
        stds = [summary[m]["metrics"].get(key, {}).get("std", 0) for m in models]
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=metric_pretty[key], color=colors[i],
               edgecolor="black", linewidth=0.4,
               error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in models],
                       rotation=20, ha="right")
    ax.set_ylabel("Score (test set)")
    ax.set_title("Test-set metrics across models (mean \u00b1 std, 3 seeds)",
                 fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Figure 5: per-class F1 (HOF vs NONE) -- pulled from confusion matrices
# --------------------------------------------------------------------------
def plot_per_class_f1(summary, out_path):
    models = get_models_in_order(summary)
    if not models:
        return

    none_f1, hof_f1, labels = [], [], []
    for m in models:
        cm = summary[m]["cm_total"]
        if cm is None:
            continue
        # Classes: 0=NONE, 1=HOF (per LABEL2ID in train script)
        # F1 per class from totals
        def f1_from_cm(c, idx):
            tp = c[idx, idx]
            fp = c[:, idx].sum() - tp
            fn = c[idx, :].sum() - tp
            if tp == 0:
                return 0.0
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            return 2 * p * r / (p + r)

        none_f1.append(f1_from_cm(cm, 0))
        hof_f1.append(f1_from_cm(cm, 1))
        labels.append(PRETTY.get(m, m))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 4.5))
    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width / 2, none_f1, width, label="F1 (NONE)",
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, hof_f1, width, label="F1 (HOF)",
           color="#C44E52", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Per-class F1 (computed from totals)")
    ax.set_title("Per-class F1 on test set (aggregated across seeds)",
                 fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    for i, (n, h) in enumerate(zip(none_f1, hof_f1)):
        ax.text(i - width / 2, n + 0.012, f"{n:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    configure_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    if not runs:
        print(f"No runs found under {args.results_dir}/. Did you train any models yet?")
        return
    print(f"Loaded {len(runs)} runs from {args.results_dir}/")

    summary = aggregate_by_model(runs)
    print(f"Aggregated into {len(summary)} models:")
    for m, s in summary.items():
        f1 = s["metrics"].get("test_macro_f1", {})
        if f1:
            print(f"  {m}: macro-F1 = {f1['mean']:.4f} \u00b1 {f1['std']:.4f} "
                  f"(n={s['n_seeds']})")

    plot_confusion_matrices_per_seed(summary, out_dir / "confusion_matrices_per_seed.png")
    plot_confusion_matrices_aggregated(summary, out_dir / "confusion_matrices_aggregated.png")
    plot_macro_f1(summary, out_dir / "macro_f1_comparison.png")
    plot_all_metrics(summary, out_dir / "all_metrics_comparison.png")
    plot_per_class_f1(summary, out_dir / "per_class_f1.png")

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
