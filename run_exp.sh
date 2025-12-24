#!/bin/bash
# Comparative Experiment Script for ReChorus
# Run SelfGNN, SASRec, LightGCN on multiple datasets

set -e  # Exit on error

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Common hyperparameters
EPOCH=20
BATCH_SIZE=256
NUM_WORKERS=0
GPU=0

# Models to compare
MODELS=("SelfGNN" "SASRec" "LightGCN")

# Datasets to evaluate on
DATASETS=("Grocery_and_Gourmet_Food" "MovieLens_1M")

# Model-specific arguments
declare -A SELFGNN_ARGS=(
    [time_periods]=6
    [gnn_layer]=2
    [att_layer]=2
    [emb_size]=64
    [num_layers]=1
    [num_heads]=4
    [ssl_weight]=1e-6
)

declare -A SASREC_ARGS=(
    [emb_size]=64
    [num_layers]=2
    [num_heads]=4
)

declare -A LIGHTGCN_ARGS=(
    [emb_size]=64
    [num_layers]=3
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to build command
build_command() {
    local model=$1
    local dataset=$2
    local cmd="python $REPO_ROOT/src/main.py \
        --model_name $model \
        --dataset $dataset \
        --epoch $EPOCH \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --gpu $GPU"
    
    case $model in
        SelfGNN)
            for key in "${!SELFGNN_ARGS[@]}"; do
                cmd="$cmd --$key ${SELFGNN_ARGS[$key]}"
            done
            ;;
        SASRec)
            for key in "${!SASREC_ARGS[@]}"; do
                cmd="$cmd --$key ${SASREC_ARGS[$key]}"
            done
            ;;
        LightGCN)
            for key in "${!LIGHTGCN_ARGS[@]}"; do
                cmd="$cmd --$key ${LIGHTGCN_ARGS[$key]}"
            done
            ;;
    esac
    
    echo "$cmd"
}

# Function to run experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local cmd=$3
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/${model}_${dataset}_${timestamp}.log"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}Running: $model on $dataset${NC}"
    echo -e "${YELLOW}Log: $log_file${NC}"
    echo -e "${YELLOW}================================${NC}\n"
    
    # Run and save log
    if eval "$cmd" | tee "$log_file"; then
        echo -e "${GREEN}✓ $model on $dataset PASSED${NC}\n"
        return 0
    else
        echo -e "${RED}✗ $model on $dataset FAILED${NC}\n"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}ReChorus Comparative Experiment${NC}"
    echo -e "${GREEN}Models: ${MODELS[@]}${NC}"
    echo -e "${GREEN}Datasets: ${DATASETS[@]}${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    local total=0
    local success=0
    local start_time=$(date +%s)
    
    # Create summary file
    local summary_file="$LOG_DIR/experiment_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Experiment Summary"
        echo "Start time: $(date)"
        echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]}))"
        echo "========================================"
    } > "$summary_file"
    
    # Run experiments
    for dataset in "${DATASETS[@]}"; do
        echo "========================================"
        echo "Dataset: $dataset"
        echo "========================================" | tee -a "$summary_file"
        
        for model in "${MODELS[@]}"; do
            ((total++))
            cmd=$(build_command "$model" "$dataset")
            
            if run_experiment "$model" "$dataset" "$cmd"; then
                ((success++))
                echo "$model: SUCCESS" >> "$summary_file"
            else
                echo "$model: FAILED" >> "$summary_file"
            fi
        done
    done
    
    # Print summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Experiment Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Total: $total | Successful: $success | Failed: $((total - success))"
    echo "Duration: $((duration / 60)) min $((duration % 60)) sec"
    echo "Summary: $summary_file"
    echo "Logs: $LOG_DIR"
    echo -e "${GREEN}========================================${NC}\n"
    
    return $((total - success))
}

# Run main
main "$@"
