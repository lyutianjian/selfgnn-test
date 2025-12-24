# ReChorus Comparative Experiment Guide

## Overview

This guide explains how to run comparative experiments for sequential recommendation models using the ReChorus framework.

**Models compared:**
- **SelfGNN**: Your new model combining graph neural networks and self-attention
- **SASRec**: Self-Attentive Sequential Recommendation (baseline)
- **LightGCN**: Light Graph Convolutional Network (baseline)

**Datasets:**
- Grocery_and_Gourmet_Food (Amazon reviews)
- MovieLens_1M (Movie ratings)

---

## Quick Start

### 1. Run All Experiments (Recommended)

**On Windows (PowerShell):**
```powershell
python run_exp.py
```

**On Linux/Mac (Bash):**
```bash
chmod +x run_exp.sh
./run_exp.sh
```

### 2. Run Specific Model on Specific Dataset

```bash
python src/main.py \
  --model_name SelfGNN \
  --dataset Grocery_and_Gourmet_Food \
  --epoch 20 \
  --batch_size 256 \
  --num_workers 0
```

Replace `SelfGNN` with `SASRec` or `LightGCN` for other models.

### 3. Analyze Results

After experiments complete:
```python
python analyze_results.py
```

This will:
- Parse all log files
- Create comparison tables
- Export detailed results to JSON

---

## Directory Structure

```
ReChorus/
├── run_exp.py              # Main experiment script (Python)
├── run_exp.sh              # Alternative script (Bash)
├── analyze_results.py      # Results analysis script
├── logs/                   # Experiment logs (auto-created)
│   ├── SelfGNN_Grocery_*_*.log
│   ├── SASRec_Grocery_*_*.log
│   ├── LightGCN_Grocery_*_*.log
│   ├── SelfGNN_MovieLens_*_*.log
│   ├── ...
│   └── experiment_summary_*.txt
├── model/                  # Saved model weights
├── log/                    # Final recommendations
└── src/
    └── main.py             # Main training script
```

---

## Hyperparameter Configuration

### Common Parameters (for all models)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `epoch` | 20 | Number of training epochs |
| `batch_size` | 256 | Batch size for training |
| `num_workers` | 0 | DataLoader workers (0 = no multiprocessing) |
| `gpu` | 0 | GPU device ID |

### Model-Specific Parameters

#### SelfGNN
```
--time_periods 6         # Number of time-based graphs
--gnn_layer 2            # GNN message passing layers
--att_layer 2            # Sequence attention layers
--emb_size 64            # Embedding dimension
--num_layers 1           # GRU layers
--num_heads 4            # Attention heads
--ssl_weight 1e-6        # SSL loss weight
```

#### SASRec
```
--emb_size 64            # Embedding dimension
--num_layers 2           # Transformer layers
--num_heads 4            # Attention heads
```

#### LightGCN
```
--emb_size 64            # Embedding dimension
--num_layers 3           # GCN layers
```

---

## Understanding Log Files

Each experiment generates a log file with this naming pattern:
```
{MODEL}_{DATASET}_{TIMESTAMP}.log
```

Example: `SelfGNN_Grocery_and_Gourmet_Food_20251224_151131.log`

### Key Metrics in Logs

Look for lines like:
```
Test After Training: (HR@5:0.3057,NDCG@5:0.1999,HR@10:0.4327,NDCG@10:0.2410,...)
```

**Metric Explanation:**
- **HR@K** (Hit Rate): Percentage of test cases with correct recommendation in top-K
- **NDCG@K** (Normalized Discounted Cumulative Gain): Ranking quality metric considering position

Higher is better for all metrics.

---

## Expected Runtime

| Dataset | Epoch | Batch Size | Time per Model |
|---------|-------|-----------|-----------------|
| Grocery_and_Gourmet_Food | 20 | 256 | ~2-3 hours |
| MovieLens_1M | 20 | 256 | ~1-2 hours |

**Total time for full experiment: ~6-10 hours**

To speed up: reduce `epoch` to 5-10, or use smaller `batch_size`.

---

## Troubleshooting

### Error: "CUDA out of memory"
Reduce batch size:
```bash
python run_exp.py --batch_size 128
```

### Error: "Cannot serialize tensor"
Ensure `--num_workers 0` is set. This is the default in the scripts.

### Experiment interrupted?
Simply run the script again. It will:
- Skip already completed runs (different timestamps)
- Continue with remaining models/datasets

### MovieLens_1M not found?
Need to download the data first:
```bash
cd data/MovieLens_1M
# Run the notebook or download script
python MovieLens-1M.ipynb
```

---

## Customizing Experiments

### Run only specific models
Edit `run_exp.py` line 24:
```python
MODELS = ["SelfGNN", "SASRec"]  # Remove LightGCN
```

### Run only specific datasets
Edit `run_exp.py` line 28:
```python
DATASETS = ["Grocery_and_Gourmet_Food"]  # Use only one dataset
```

### Change hyperparameters
Edit the dictionaries in `run_exp.py`:
```python
COMMON_ARGS = {
    "epoch": 50,          # Increase epochs
    "batch_size": 512,    # Larger batches
    ...
}

MODEL_ARGS = {
    "SelfGNN": {
        "emb_size": 128,   # Larger embeddings
        ...
    },
    ...
}
```

---

## Analyzing Results

### 1. View Logs in Real-Time
```bash
tail -f logs/SelfGNN_*.log
```

### 2. Extract Metrics
```python
python analyze_results.py
```

This creates:
- **detailed_results.json**: Complete metrics for all models/datasets
- **Console output**: Comparison tables

### 3. Manual Comparison
```bash
grep "Test After Training" logs/*.log
```

---

## Example Output

```
====================================================================
EXPERIMENTAL RESULTS COMPARISON
====================================================================

Dataset: Grocery_and_Gourmet_Food
--------------------------------------------------------------------
            test_HR@5  test_NDCG@5  test_HR@10  test_NDCG@10  test_HR@20  test_NDCG@20
SelfGNN       0.3057      0.1999      0.4327      0.2410      0.5570      0.2724
SASRec        0.2918      0.1891      0.4102      0.2287      0.5302      0.2604
LightGCN      0.2456      0.1543      0.3654      0.1974      0.4891      0.2234

Dataset: MovieLens_1M
--------------------------------------------------------------------
            test_HR@5  test_NDCG@5  test_HR@10  test_NDCG@10  test_HR@20  test_NDCG@20
SelfGNN       0.3245      0.2156      0.4512      0.2634      0.5823      0.2987
SASRec        0.3102      0.2045      0.4345      0.2501      0.5621      0.2856
LightGCN      0.2734      0.1756      0.3987      0.2234      0.5234      0.2567
```

---

## Tips for Writing Assignment Report

1. **Model Comparison**: Include the comparison tables from `analyze_results.py`
2. **Statistical Significance**: Run 3 times and report average ± std
3. **Ablation Study**: Try disabling components of SelfGNN:
   - Remove SSL loss: `--ssl_weight 0`
   - Change time_periods: `--time_periods 3,6,12`
4. **Runtime Analysis**: Include time per epoch
5. **Discussion**: Explain why SelfGNN performs better/worse

---

## Additional Resources

- **SASRec Paper**: https://arxiv.org/abs/1809.03201
- **LightGCN Paper**: https://arxiv.org/abs/2002.02126
- **ReChorus Framework**: Check `src/README.md`

---

## Questions?

Check the log files for detailed error messages:
```bash
cat logs/experiment_summary_*.txt
```

Or run a single model to debug:
```bash
python src/main.py --model_name SelfGNN --dataset Grocery_and_Gourmet_Food \
  --epoch 1 --batch_size 32 --num_workers 0
```
