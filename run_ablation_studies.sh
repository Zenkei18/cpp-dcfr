#!/bin/bash
# Ablation Study Script for Neural Network Convergence Accelerator

echo "Starting Neural Network Convergence Accelerator Ablation Studies"
echo "================================================================"

# Create results directory
mkdir -p results/ablation

# Baseline configuration
echo "Running baseline configuration..."
python3 src/training/train_enhanced.py \
    --config configs/ablate/baseline.yaml \
    --iterations 1000 \
    --save-dir results/ablation/baseline \
    --log-dir logs/ablation/baseline \
    --verbose

# AdamW + Clipping configuration
echo "Running AdamW + Clipping configuration..."
python3 src/training/train_enhanced.py \
    --config configs/ablate/adamw_clip.yaml \
    --iterations 1000 \
    --save-dir results/ablation/adamw_clip \
    --log-dir logs/ablation/adamw_clip \
    --verbose

# Cosine + AMP configuration
echo "Running Cosine + AMP configuration..."
python3 src/training/train_enhanced.py \
    --config configs/ablate/cosine_amp.yaml \
    --iterations 1000 \
    --save-dir results/ablation/cosine_amp \
    --log-dir logs/ablation/cosine_amp \
    --verbose

# Target Normalization configuration
echo "Running Target Normalization configuration..."
python3 src/training/train_enhanced.py \
    --config configs/ablate/norm_targets.yaml \
    --iterations 1000 \
    --save-dir results/ablation/norm_targets \
    --log-dir logs/ablation/norm_targets \
    --verbose

echo "All ablation studies completed!"
echo "View results with: tensorboard --logdir logs/ablation"
echo "Training reports are available in results/ablation/*/docs/"