#!/bin/bash
# Run on your local machine.
# Covers: TF-IDF baseline + mBERT + XLM-R + MuRIL, 3 seeds each.
# Total time estimate: ~3-4 hours.

set -e

SEEDS=(42 123 2024)

echo "=== TF-IDF + SVM baseline ==="
for seed in "${SEEDS[@]}"; do
    python src/train_tfidf.py --seed "$seed"
done

LOCAL_MODELS=(
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
    "google/muril-base-cased"
)

for model in "${LOCAL_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=== $model seed=$seed ==="
        python src/train_transformer.py \
            --model "$model" \
            --seed "$seed" \
            --batch_size 16 \
            --grad_accum 2
        # Clear cache between runs
        python -c "import torch; torch.cuda.empty_cache()" || true
    done
done

python src/aggregate_results.py
echo ""
echo "Done. See results_summary.csv"
