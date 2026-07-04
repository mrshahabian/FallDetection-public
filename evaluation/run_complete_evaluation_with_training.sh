#!/bin/bash
# Complete evaluation script WITH training - trains models first, then evaluates
# Runs all models with 1 and 8 frames
# Logs all results to timestamped directories

source venv/bin/activate

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RESULTS_DIR="evaluation/results/complete_evaluation_${TIMESTAMP}"

echo "=========================================="
echo "Complete Evaluation WITH Training - All Models"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

# Create log file
LOG_FILE="${BASE_RESULTS_DIR}/evaluation.log"
mkdir -p "${BASE_RESULTS_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to: ${LOG_FILE}"
echo ""

# Function to train CNN model
train_cnn_model() {
    local model_type=$1
    local checkpoint_path="evaluation/checkpoints/${model_type}_fall_detection.pt"
    
    echo "=========================================="
    echo "Training ${model_type}"
    echo "=========================================="
    echo "Checkpoint will be saved to: ${checkpoint_path}"
    echo ""
    
    python3 evaluation/train_cnn_models.py \
        --model_type ${model_type} \
        --fall_base_dir "/home/reza/Documents/Datasets/Fall/Fall" \
        --kth_dir "/home/reza/Documents/Datasets/KTH" \
        --num_fall 100 \
        --num_non_fall 100 \
        --epochs 50 \
        --batch_size 8 \
        --window_length 32 \
        --overlap 0.5 \
        --save_path "${checkpoint_path}" \
        --device auto 2>&1 | tee -a "${LOG_FILE}"
    
    if [ -f "${checkpoint_path}" ]; then
        echo ""
        echo "✓ ${model_type} training completed! Checkpoint saved."
        echo ""
    else
        echo ""
        echo "⚠️  WARNING: ${model_type} checkpoint not found after training!"
        echo ""
    fi
}

# Function to train Few-Shot VLM
train_few_shot_vlm() {
    local checkpoint_path="evaluation/checkpoints/few_shot_vlm_classifier.pt"
    
    echo "=========================================="
    echo "Training Few-Shot VLM Classifier"
    echo "=========================================="
    echo "Checkpoint will be saved to: ${checkpoint_path}"
    echo ""
    
    python3 evaluation/train_few_shot_vlm.py \
        --fall_base_dir "/home/reza/Documents/Datasets/Fall/Fall" \
        --non_fall_videos_dir "/home/reza/Documents/Datasets/KTH" \
        --num_shots 20 \
        --seed 42 \
        --save_path "${checkpoint_path}" 2>&1 | tee -a "${LOG_FILE}"
    
    if [ -f "${checkpoint_path}" ]; then
        echo ""
        echo "✓ Few-Shot VLM training completed! Checkpoint saved."
        echo ""
    else
        echo ""
        echo "⚠️  WARNING: Few-Shot VLM checkpoint not found after training!"
        echo ""
    fi
}

# Function to run VLM evaluation with specific frame count
run_vlm_evaluation() {
    local num_frames=$1
    local results_dir="${BASE_RESULTS_DIR}/vlm_${num_frames}frames"
    
    echo "=========================================="
    echo "Running VLM Evaluation with ${num_frames} frames"
    echo "=========================================="
    echo "Results directory: ${results_dir}"
    echo ""
    
    python3 evaluation/evaluate_vlm_comprehensive.py \
        --fall_base_dir "/home/reza/Documents/Datasets/Fall/Fall" \
        --kth_dir "/home/reza/Documents/Datasets/KTH" \
        --num_samples 100 \
        --shared_test_set "${SHARED_TEST_SET}" \
        --few_shot_classifier "evaluation/checkpoints/few_shot_vlm_classifier.pt" \
        --seed 42 \
        --window_length 32 \
        --overlap 0.5 \
        --num_frames ${num_frames} \
        --device auto \
        --results_dir "${results_dir}" 2>&1 | tee -a "${LOG_FILE}"
    
    echo ""
    echo "VLM Evaluation (${num_frames} frames) completed!"
    echo ""
}

# Function to run supervised methods evaluation
run_supervised_evaluation() {
    local suffix=$1
    local results_dir="${BASE_RESULTS_DIR}/supervised_${suffix}"
    
    echo "=========================================="
    echo "Running Supervised Methods Evaluation (${suffix})"
    echo "=========================================="
    echo "Results directory: ${results_dir}"
    echo ""
    
    # Single-person dataset only (best fitted process)
    # Use shared test set for fair comparison
    python3 evaluation/evaluate_all.py \
        --shared_test_set "${SHARED_TEST_SET}" \
        --results_dir "${results_dir}" \
        --cnn_checkpoints \
            "2dcnn_resnet:evaluation/checkpoints/2dcnn_resnet_fall_detection.pt" \
            "3dcnn_simple:evaluation/checkpoints/3dcnn_simple_fall_detection.pt" \
            "vit:evaluation/checkpoints/vit_fall_detection.pt" \
        --window_length 32 \
        --overlap 0.5 \
        --device auto 2>&1 | tee -a "${LOG_FILE}"
    
    echo ""
    echo "Supervised Methods Evaluation (${suffix}) completed!"
    echo ""
}

# ============================================================================
# Training Phase
# ============================================================================

echo "=========================================="
echo "PHASE 1: TRAINING MODELS"
echo "=========================================="
echo "Start time: $(date)"
echo ""

    # Check if checkpoints exist, ask user if they want to retrain
SKIP_TRAINING=false
if [ -f "evaluation/checkpoints/2dcnn_resnet_fall_detection.pt" ] && \
   [ -f "evaluation/checkpoints/3dcnn_simple_fall_detection.pt" ] && \
   [ -f "evaluation/checkpoints/vit_fall_detection.pt" ] && \
   [ -f "evaluation/checkpoints/few_shot_vlm_classifier.pt" ]; then
    echo "Existing checkpoints found. Options:"
    echo "  1. Skip training (use existing checkpoints)"
    echo "  2. Retrain all models"
    echo ""
    read -p "Enter choice (1 or 2, default=1): " choice
    if [ "$choice" == "1" ] || [ -z "$choice" ]; then
        SKIP_TRAINING=true
        echo "Skipping training, using existing checkpoints."
    else
        echo "Retraining all models..."
    fi
fi

if [ "$SKIP_TRAINING" = false ]; then
    # Create checkpoints directory
    mkdir -p evaluation/checkpoints
    
    # Train models that need retraining
    echo ""
    echo "Training models that showed poor performance..."
    echo ""
    
    # Train 3D CNN
    train_cnn_model "3dcnn_simple"
    
    # Train ViT
    train_cnn_model "vit"
    
    # Train Few-Shot VLM
    train_few_shot_vlm
    
    echo ""
    echo "Training phase completed!"
    echo ""
else
    echo "Using existing checkpoints (training skipped)."
    echo ""
fi

# ============================================================================
# Create Shared Test Set
# ============================================================================

echo "=========================================="
echo "Creating Shared Test Set"
echo "=========================================="
echo "This ensures all models use the same test samples for fair comparison"
echo ""

SHARED_TEST_SET="evaluation/data/shared_test_set.txt"
python3 evaluation/create_shared_test_set.py \
    --fall_base_dir "/home/reza/Documents/Datasets/Fall/Fall" \
    --kth_dir "/home/reza/Documents/Datasets/KTH" \
    --num_samples 100 \
    --seed 42 \
    --output_file "${SHARED_TEST_SET}" 2>&1 | tee -a "${LOG_FILE}"

if [ -f "${SHARED_TEST_SET}" ]; then
    echo ""
    echo "✓ Shared test set created: ${SHARED_TEST_SET}"
    echo ""
else
    echo ""
    echo "⚠️  WARNING: Shared test set not created. Each model will sample independently."
    echo ""
fi

# ============================================================================
# Evaluation Phase
# ============================================================================

echo "=========================================="
echo "PHASE 2: EVALUATION"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# 1. VLM with 1 frame
run_vlm_evaluation 1

# 2. VLM with 8 frames
run_vlm_evaluation 8

# 3. Supervised methods (Single-Person Only)
run_supervised_evaluation "single_person"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Complete Evaluation WITH Training - Finished!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: ${BASE_RESULTS_DIR}"
echo ""
echo "Directory structure:"
echo "  ${BASE_RESULTS_DIR}/"
echo "    ├── vlm_1frames/"
echo "    ├── vlm_8frames/"
echo "    ├── supervised_single_person/"
echo "    └── evaluation.log"
echo ""
echo "Checkpoints saved to: evaluation/checkpoints/"
echo ""
echo "To view results:"
echo "  cat ${BASE_RESULTS_DIR}/evaluation.log"
echo "  ls -lh ${BASE_RESULTS_DIR}/*/comparison_table*.txt"
echo ""

