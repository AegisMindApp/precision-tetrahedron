#!/bin/bash
# FlashOptim TPU experiment — master runner
# Runs conditions A → B → C sequentially.
# Do NOT run until Google confirms TPU project registration.

set -e

DATA_DIR="/tmp/qm9"
OUTPUT_DIR="/tmp/flashoptim_results"
EPOCHS=100
BATCH_SIZE=32
LR=1e-4

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "FlashOptim Mixed-Precision TPU Experiment"
echo "========================================"
echo "Data:    $DATA_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS"
echo ""

# Condition A — FP32 baseline
echo ">>> Condition A: FP32 baseline"
python3 train.py \
    --condition A \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo ">>> Condition B: BF16 mixed-precision (same size)"
python3 train.py \
    --condition B \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo ">>> Condition C: BF16 mixed-precision (2x wider)"
python3 train.py \
    --condition C \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo ""
echo ">>> Comparing results..."
python3 compare.py --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Done. Results in $OUTPUT_DIR ==="
