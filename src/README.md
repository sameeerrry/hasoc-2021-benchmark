# HASOC 2021 Subtask 1 Benchmark

Binary hate/offensive vs non-hate classification for code-mixed Hindi-English tweets.

## Setup

```bash
pip install -r requirements.txt
```

Place the raw data like this:

```
hasoc2021/
├── data/
│   └── raw/
│       └── train/
│           ├── bantwitter/
│           │   ├── 1397136107108663298/
│           │   │   ├── data.json
│           │   │   └── labels.json
│           │   └── ...
│           ├── casteism/
│           └── ...
│       └── test/        
│           ├── bantwitter/
│           │   ├── 1397136107108663298/
│           │   │   ├── data.json
│           │   │   └── labels.json
│           │   └── ...
│           ├── casteism/
│           └── ...
```

## Pipeline

```bash
# 1. Flatten folder tree -> single CSV
#    (add --test_dir if you have an official test set)
python src/load_data.py --train_dir data/raw/train

# 2. Clean + split by conversation (prevents leakage)
python src/split_data.py --input data/processed/full_train.csv --seed 42
#    If you already have an official test set:
# python src/split_data.py --input data/processed/full_train.csv --seed 42 --has_official_test

# 3. Train baseline
python src/train_tfidf.py --seed 42

# 4. Train transformers
python src/train_transformer.py --model xlm-roberta-base --seed 42

# 5. Aggregate all runs into paper-ready table
python src/aggregate_results.py
```

## Running the full benchmark

```bash
# Local GPU: TF-IDF + mBERT + XLM-R + MuRIL
bash run_local.sh

# Colab T4 (in parallel): IndicBERT + HingRoBERTa
# see colab_runner.py
```

## Output

- `results/{model_tag}_seed{N}/metrics.json` — per-run metrics
- `results/{model_tag}_seed{N}/classification_report.txt` — per-class P/R/F1
- `results/{model_tag}_seed{N}/confusion_matrix.npy`
- `results/{model_tag}_seed{N}/predictions.csv`
- `results_summary.csv` — aggregated mean±std (paper-ready)
- `results_summary_latex.tex` — LaTeX table snippet
