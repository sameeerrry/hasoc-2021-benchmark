"""
Split the full training DataFrame into train/val/test.

Critical: splits by conversation_id, not by tweet_id, to prevent leakage
(tweets from the same conversation reference each other and share context).

Usage:
    python src/split_data.py --input data/processed/full_train.csv \\
                             --output_dir data/processed \\
                             --seed 42
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from clean import clean_tweet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10,
                    help="Ignored if a separate official test set already exists")
    ap.add_argument("--has_official_test", action="store_true",
                    help="Set if test.csv already exists; only splits train/val")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['conversation_id'].nunique()} conversations")

    # Clean text
    print("Cleaning text...")
    df["text"] = df["text"].astype(str).apply(clean_tweet)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"After dropping empty: {len(df)} rows")

    # Drop ultra-short tweets (single emoji, single word)
    df = df[df["text"].str.split().str.len() >= 2].reset_index(drop=True)
    print(f"After min-2-tokens filter: {len(df)} rows")

    # Conversation-level split, stratified by (topic, hate-ratio-bucket).
    #
    # With only ~82 conversations AND wildly varying hate ratios per conversation
    # (some threads 10% HOF, others 80% HOF), stratifying only by topic leaves
    # label balance to chance. A bad draw yields e.g. train=54% HOF but test=69%,
    # which biases all downstream comparisons. We therefore also bucket each
    # conversation into low/mid/high HOF-ratio and stratify on (topic, bucket).
    conv_to_topic = (
        df[["conversation_id", "topic"]]
        .drop_duplicates()
        .set_index("conversation_id")["topic"]
        .to_dict()
    )

    # Compute each conversation's HOF ratio, bucket into thirds (L/M/H).
    conv_hof_ratio = (
        df.assign(is_hof=(df["label"] == "HOF").astype(int))
          .groupby("conversation_id")["is_hof"].mean()
    )
    # Bucket by terciles of the observed distribution
    low_q, high_q = conv_hof_ratio.quantile([0.33, 0.67]).values

    def bucket(r):
        if r <= low_q:  return "L"
        if r <= high_q: return "M"
        return "H"

    conv_to_bucket = conv_hof_ratio.apply(bucket).to_dict()

    rng = np.random.RandomState(args.seed)

    # Group conversations by (topic, bucket) and shuffle within each.
    from collections import defaultdict
    by_stratum = defaultdict(list)
    for cid, topic in conv_to_topic.items():
        by_stratum[(topic, conv_to_bucket[cid])].append(cid)
    for key in by_stratum:
        rng.shuffle(by_stratum[key])

    n_total = sum(len(v) for v in by_stratum.values())

    # Target global fold sizes (in conversations).
    if args.has_official_test:
        # Scale val_frac to compensate for test being absent.
        target_val = int(round(n_total * args.val_frac / (1 - args.test_frac)))
        target_test = 0
    else:
        target_val = int(round(n_total * args.val_frac))
        target_test = int(round(n_total * args.test_frac))
    target_train = n_total - target_val - target_test

    # Per-stratum targets via largest-remainder rounding.
    # For each fold, each stratum's ideal share is n * frac; floor each,
    # then hand out the remaining slots to strata with the largest fractional
    # remainders. This makes global sums exact and spreads allocation fairly
    # across strata (so small topics do end up in val/test, not always train).
    def allocate(stratum_sizes, total_target):
        if total_target == 0 or n_total == 0:
            return {k: 0 for k in stratum_sizes}
        raw = {k: n * total_target / n_total for k, n in stratum_sizes.items()}
        floored = {k: int(v) for k, v in raw.items()}
        leftover = total_target - sum(floored.values())
        # Sort strata by descending remainder, break ties by stratum key for determinism
        remainders = sorted(
            raw.keys(),
            key=lambda k: (-(raw[k] - floored[k]), str(k)),
        )
        for k in remainders[:leftover]:
            floored[k] += 1
        return floored

    stratum_sizes = {k: len(v) for k, v in by_stratum.items()}
    per_stratum_test = allocate(stratum_sizes, target_test)
    per_stratum_val = allocate(stratum_sizes, target_val)

    # Safety: if any stratum would lose all its convs to val+test, clip back.
    for k, n in stratum_sizes.items():
        if per_stratum_test[k] + per_stratum_val[k] >= n:
            # Keep at least 1 in train; prefer giving up test first, then val.
            over = per_stratum_test[k] + per_stratum_val[k] - (n - 1)
            take_from_test = min(over, per_stratum_test[k])
            per_stratum_test[k] -= take_from_test
            over -= take_from_test
            per_stratum_val[k] -= over  # remaining, guaranteed <= per_stratum_val[k]

    train_convs, val_convs, test_convs = set(), set(), set()
    for k, cids in by_stratum.items():
        nt = per_stratum_test[k]
        nv = per_stratum_val[k]
        test_convs.update(cids[:nt])
        val_convs.update(cids[nt:nt + nv])
        train_convs.update(cids[nt + nv:])

    train_df = df[df["conversation_id"].isin(train_convs)].reset_index(drop=True)
    val_df = df[df["conversation_id"].isin(val_convs)].reset_index(drop=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    print(f"\nTrain: {len(train_df)} rows ({len(train_convs)} convs)")
    print(train_df["label"].value_counts())
    print(f"\nVal: {len(val_df)} rows ({len(val_convs)} convs)")
    print(val_df["label"].value_counts())

    if not args.has_official_test:
        test_df = df[df["conversation_id"].isin(test_convs)].reset_index(drop=True)
        test_df.to_csv(out_dir / "test.csv", index=False)
        print(f"\nTest: {len(test_df)} rows ({len(test_convs)} convs)")
        print(test_df["label"].value_counts())

    # Per-topic breakdown (sanity check)
    print("\n=== Per-topic split (conversations) ===")
    split_map = {}
    for c in train_convs: split_map[c] = "train"
    for c in val_convs:   split_map[c] = "val"
    for c in test_convs:  split_map[c] = "test"

    topic_split = pd.DataFrame([
        {"conversation_id": c, "topic": conv_to_topic[c], "split": split_map[c]}
        for c in split_map
    ])
    print(topic_split.groupby(["topic", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
