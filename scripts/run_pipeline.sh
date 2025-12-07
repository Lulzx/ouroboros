#!/bin/bash
# Run the full GRN generation training pipeline

set -e

echo "=================================================="
echo "GRN Generation with Self-Classifying GRPO"
echo "=================================================="

# Step 1: Preprocess data
echo ""
echo "Step 1: Preprocessing data..."
echo "=================================================="
python scripts/preprocess.py \
    --circuits data/synthetic/classic_circuits.json \
    --output-dir data/processed \
    --expand 10 \
    --seed 42

# Step 2: Supervised pretraining
echo ""
echo "Step 2: Supervised pretraining..."
echo "=================================================="
python scripts/train_supervised.py \
    --data-dir data/processed \
    --checkpoint-dir checkpoints/supervised \
    --epochs 20 \
    --batch-size 32 \
    --augment \
    --log-interval 10

# Step 3: GRPO training with self-classification
echo ""
echo "Step 3: GRPO training (self-classification)..."
echo "=================================================="
python scripts/train_grpo.py \
    --data-dir data/processed \
    --supervised-checkpoint checkpoints/supervised/best.safetensors \
    --checkpoint-dir checkpoints/grpo \
    --steps 2000 \
    --log-interval 10 \
    --eval-interval 100

# Step 4: Oracle-augmented GRPO (optional)
echo ""
echo "Step 4: Oracle GRPO training..."
echo "=================================================="
python scripts/train_oracle_grpo.py \
    --data-dir data/processed \
    --checkpoint checkpoints/grpo/checkpoint_step2000.safetensors \
    --checkpoint-dir checkpoints/oracle_grpo \
    --steps 1000 \
    --self-class-weight 0.5 \
    --oracle-weight 0.5 \
    --log-interval 10 \
    --eval-interval 100

# Step 5: Evaluation
echo ""
echo "Step 5: Evaluating models..."
echo "=================================================="

echo "Evaluating supervised model..."
python scripts/evaluate.py \
    --checkpoint checkpoints/supervised/best.safetensors \
    --data-dir data/processed \
    --num-samples 50 \
    --output results/eval_supervised.json

echo "Evaluating GRPO model..."
python scripts/evaluate.py \
    --checkpoint checkpoints/grpo/checkpoint_step2000.safetensors \
    --data-dir data/processed \
    --num-samples 50 \
    --output results/eval_grpo.json

echo "Evaluating Oracle GRPO model..."
python scripts/evaluate.py \
    --checkpoint checkpoints/oracle_grpo/checkpoint_step1000.safetensors \
    --data-dir data/processed \
    --num-samples 50 \
    --output results/eval_oracle_grpo.json

# Step 6: Generate sample circuits
echo ""
echo "Step 6: Generating sample circuits..."
echo "=================================================="
python scripts/generate.py \
    --checkpoint checkpoints/oracle_grpo/checkpoint_step1000.safetensors \
    --data-dir data/processed \
    --all-phenotypes \
    --num-samples 10 \
    --simulate \
    --output results/generated_circuits.json

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "Results saved to results/"
echo "=================================================="
