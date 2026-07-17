#!/bin/bash

# Experiment: Testing different values for loss_w parameter
# We will test w = 0, 1, 10, 30, 100
# Keeping other parameters constant: max_jitter=32, batch_size=128

WEIGHTS=(0.0 1.0 10.0 30.0 100.0)
MAX_JITTER=32
BATCH_SIZE=128
OUTPUT_PATH="./hyperparameter_search"
DATASET_ID="RBFOX2_HepG2_ENCSR987FTF"

# Ensure the output directory exists
mkdir -p "$OUTPUT_PATH"

echo "Starting hyperparameter search for w parameter..."

for W in "${WEIGHTS[@]}"; do
    # Format w to drop trailing decimal 0 for cleaner naming if desired, or keep it.
    EXPERIMENT_ID="w_${W}_jitter_${MAX_JITTER}"

    echo "=========================================================="
    echo "Running experiment: $EXPERIMENT_ID with w=$W"
    echo "=========================================================="

    uv run python train_from_hf.py \
        --experiment_id "$EXPERIMENT_ID" \
        --dataset_id "$DATASET_ID" \
        --max_jitter $MAX_JITTER \
        --loss_w "$W" \
        --batch_size $BATCH_SIZE \
        --output_path "$OUTPUT_PATH"

    echo "Finished experiment: $EXPERIMENT_ID"
    echo ""
done

echo "Hyperparameter search complete!"