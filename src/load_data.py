"""
Walk the HASOC 2021 Subtask 1 folder tree and build a unified DataFrame.

Expected structure:
    <root>/
        <topic>/
            <conversation_id>/
                data.json       # {tweet_id, tweet, comments: [{tweet_id, tweet}, ...]}
                labels.json     # {tweet_id: "HOF" | "NONE", ...}

Output:
    CSV with columns: tweet_id, text, label, topic, conversation_id, is_root
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def load_conversation(conv_dir: Path) -> list:
    """Load a single conversation into a list of rows."""
    data_path = conv_dir / "data.json"
    labels_path = conv_dir / "labels.json"

    if not (data_path.exists() and labels_path.exists()):
        return []

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARN] Skipping {conv_dir}: {e}")
        return []

    rows = []
    conv_id = conv_dir.name
    topic = conv_dir.parent.name

    # Root tweet
    root_id = data.get("tweet_id")
    root_text = data.get("tweet")
    if root_id and root_id in labels and root_text:
        rows.append({
            "tweet_id": str(root_id),
            "text": root_text,
            "label": labels[root_id],
            "topic": topic,
            "conversation_id": conv_id,
            "is_root": True,
        })

    # Comments
    for c in data.get("comments", []):
        cid = c.get("tweet_id")
        ctext = c.get("tweet")
        if cid and cid in labels and ctext:
            rows.append({
                "tweet_id": str(cid),
                "text": ctext,
                "label": labels[cid],
                "topic": topic,
                "conversation_id": conv_id,
                "is_root": False,
            })

    return rows


def load_split(split_root: Path) -> pd.DataFrame:
    """Load a single split (train/val/test) into a DataFrame."""
    if not split_root.exists():
        raise FileNotFoundError(
            f"Path does not exist: {split_root.resolve()}\n"
            f"  Check that --train_dir points to the folder containing your topic subfolders.\n"
            f"  Example layout:\n"
            f"    {split_root}/bantwitter/<conv_id>/data.json\n"
            f"    {split_root}/casteism/<conv_id>/data.json"
        )

    conv_dirs = list(split_root.rglob("data.json"))
    if not conv_dirs:
        # No data.json files found
        children = [p.name for p in split_root.iterdir() if p.is_dir()][:10]
        raise FileNotFoundError(
            f"No data.json files found anywhere under: {split_root.resolve()}\n"
            f"  Subdirectories present: {children}\n"
            f"  Expected structure:  <train_dir>/<topic>/<conversation_id>/data.json\n"
            f"  Did you mean a different --train_dir? Try searching manually with:\n"
            f"    Get-ChildItem -Path {split_root} -Recurse -Filter data.json"
        )

    print(f"Found {len(conv_dirs)} conversation folders")
    all_rows = []
    for data_json in tqdm(conv_dirs, desc=f"Loading {split_root.name}"):
        rows = load_conversation(data_json.parent)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError(
            f"Found {len(conv_dirs)} data.json files but extracted zero labeled rows.\n"
            f"  This usually means labels.json is missing or its tweet_ids don't match data.json."
        )

    df = pd.DataFrame(all_rows)

    # Normalize labels: HOF -> HOF, NOT -> NONE
    df["label"] = df["label"].replace({"NOT": "NONE"})

    # Drop duplicates (a tweet might appear in multiple conversations)
    before = len(df)
    df = df.drop_duplicates(subset="tweet_id").reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} duplicate tweet_ids")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True,
                    help="Path to training data root (contains topic folders)")
    ap.add_argument("--test_dir", default=None,
                    help="Path to test data root; optional")
    ap.add_argument("--output_dir", default="data/processed")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(Path(args.train_dir))
    print(f"\nTrain: {len(train_df)} rows")
    print(train_df["label"].value_counts())
    print(f"Topics: {train_df['topic'].unique().tolist()}")
    print(f"Conversations: {train_df['conversation_id'].nunique()}")

    train_df.to_csv(out_dir / "full_train.csv", index=False)
    print(f"Saved: {out_dir / 'full_train.csv'}")

    if args.test_dir:
        test_df = load_split(Path(args.test_dir))
        print(f"\nTest: {len(test_df)} rows")
        print(test_df["label"].value_counts())
        test_df.to_csv(out_dir / "test.csv", index=False)
        print(f"Saved: {out_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
