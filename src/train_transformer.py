"""
Unified fine-tuning script for transformer models on HASOC 2021 Subtask 1.

Usage:
    python src/train_transformer.py --model xlm-roberta-base --seed 42
    python src/train_transformer.py --model google/muril-base-cased --seed 42
    python src/train_transformer.py --model ai4bharat/IndicBERTv2-MLM-only --seed 42
    python src/train_transformer.py --model l3cube-pune/hing-roberta --seed 42
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback,
    DataCollatorWithPadding,
)

LABEL2ID = {"NONE": 0, "HOF": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "macro_f1": f1,
        "macro_precision": p,
        "macro_recall": r,
        "binary_f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy."""
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_csv", default="data/processed/train.csv")
    ap.add_argument("--val_csv", default="data/processed/val.csv")
    ap.add_argument("--test_csv", default="data/processed/test.csv")
    ap.add_argument("--output_root", default="results")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="Use >1 to simulate larger batch on small GPUs")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_tag = args.model.replace("/", "_")
    out_dir = Path(args.output_root) / f"{model_tag}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].map(LABEL2ID)

    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # Class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Tokenizer + tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tok, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tok, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    # Custom callback: track per-epoch val macro-F1, remember best checkpoint path.
    # We do NOT use HF's load_best_model_at_end + EarlyStoppingCallback combo:
    # those rely on metric_for_best_model resolution which is broken across
    # transformers versions (prefix handling flipped multiple times between
    # 4.35, 4.45, 4.55+, with inconsistent behavior even within a version).
    # Tracking manually sidesteps all of that.
    class BestModelTracker(TrainerCallback):
        def __init__(self):
            self.best_f1 = -1.0
            self.best_epoch = None
            self.best_ckpt_dir = None
            self.history = []

        def on_evaluate(self, _args, state, control, metrics=None, **kw):
            if not metrics:
                return
            f1 = metrics.get("eval_macro_f1")
            if f1 is None:
                return
            ep = int(round(state.epoch)) if state.epoch else len(self.history) + 1
            self.history.append((ep, f1))
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = ep
                # HF saves as checkpoint-<global_step>; track the latest checkpoint dir
                # after this eval, which corresponds to this epoch's save.
                self.best_step = state.global_step

    best_tracker = BestModelTracker()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,          # we load best manually
        save_total_limit=args.epochs,          # keep all epoch checkpoints for selection
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        fp16=(device == "cuda") and not args.no_fp16,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[best_tracker],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Print training curve so we can always see what happened.
    print("\n=== Val macro-F1 per epoch ===")
    for ep, f1 in best_tracker.history:
        marker = "  <-- best" if ep == best_tracker.best_epoch else ""
        print(f"  Epoch {ep}: {f1:.4f}{marker}")

    # Load best checkpoint from disk and evaluate with a plain PyTorch loop.
    # We DO NOT go through trainer.evaluate()/predict() here: reassigning
    # trainer.model doesn't always propagate to the accelerator-wrapped model
    # that trainer internals actually use, so trainer.predict() can silently
    # run inference with the final-epoch model even after reassignment.
    ckpt_root = out_dir / "checkpoints"
    if best_tracker.best_epoch is not None:
        best_ckpt = ckpt_root / f"checkpoint-{best_tracker.best_step}"
        if not best_ckpt.exists():
            print(f"[WARN] Best checkpoint {best_ckpt} missing; falling back to final-epoch.")
            print(f"       Available: {[p.name for p in ckpt_root.iterdir()]}")
            best_ckpt = None
    else:
        best_ckpt = None

    if best_ckpt is not None:
        print(f"\nLoading best model from {best_ckpt} (val F1 = {best_tracker.best_f1:.4f})")
        eval_model = AutoModelForSequenceClassification.from_pretrained(
            best_ckpt, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ).to(device)
    else:
        eval_model = trainer.model.to(device)

    # Plain PyTorch inference loop over the test set.
    print("Running test inference with best-epoch model...")
    eval_model.eval()
    from torch.utils.data import DataLoader
    collator = DataCollatorWithPadding(tokenizer)

    # HF collator expects the label column to be called "labels". Rename,
    # and drop any non-tensor columns the model doesn't need.
    test_ds_for_eval = test_ds.rename_column("label", "labels")
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    test_ds_for_eval = test_ds_for_eval.remove_columns(
        [c for c in test_ds_for_eval.column_names if c not in keep_cols]
    )
    test_ds_for_eval.set_format("torch")

    test_loader = DataLoader(
        test_ds_for_eval,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            if device == "cuda" and not args.no_fp16:
                with torch.autocast(device_type="cuda"):
                    out = eval_model(**batch)
            else:
                out = eval_model(**batch)
            all_logits.append(out.logits.float().cpu().numpy())
            all_labels.append(labels)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    pred_labels = np.argmax(logits, axis=-1)

    # Compute all metrics.
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred_labels, average="macro", zero_division=0
    )
    test_metrics = {
        "test_macro_f1": float(f1),
        "test_macro_precision": float(p),
        "test_macro_recall": float(r),
        "test_binary_f1": float(f1_score(y_true, pred_labels, pos_label=1, zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, pred_labels, average="weighted", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, pred_labels)),
        "model": args.model,
        "seed": args.seed,
        "n_params_M": round(n_params / 1e6, 2),
        "train_time_sec": round(train_time, 1),
        "best_epoch": best_tracker.best_epoch,
        "best_val_macro_f1": round(best_tracker.best_f1, 4),
        "val_f1_per_epoch": [round(f, 4) for _, f in best_tracker.history],
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(
            y_true, pred_labels,
            target_names=["NONE", "HOF"], digits=4,
        ))

    np.save(out_dir / "confusion_matrix.npy", confusion_matrix(y_true, pred_labels))

    pd.DataFrame({
        "tweet_id": test_df["tweet_id"].values,
        "text": test_df["text"].values,
        "true": y_true,
        "pred": pred_labels,
    }).to_csv(out_dir / "predictions.csv", index=False)

    np.save(out_dir / "confusion_matrix.npy",
            confusion_matrix(test_df["label"].values, pred_labels))

    # Print results FIRST so the user always sees them, even if cleanup fails.
    print(f"\nTest macro-F1: {test_metrics['test_macro_f1']:.4f}")
    print(f"Results: {out_dir}")

    # Clean up model checkpoints (keeps disk reasonable).
    # Windows keeps file handles briefly after training; retry a few times.
    # If cleanup fails it's harmless - results are saved. Don't wait too long.
    import time as _time
    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        # Free model handles before delete. Each var may or may not exist
        # depending on which code path completed; guard each separately.
        try: del trainer
        except NameError: pass
        try: del model
        except NameError: pass
        try: del eval_model
        except NameError: pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for attempt in range(3):
            try:
                _time.sleep(0.5)
                shutil.rmtree(ckpt_dir)
                break
            except (PermissionError, OSError):
                if attempt == 2:
                    print(f"[WARN] Could not delete {ckpt_dir} (Windows file lock).")
                    print(f"       Results are saved; delete the folder manually if disk matters.")


if __name__ == "__main__":
    main()