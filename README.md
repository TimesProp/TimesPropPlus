# TimesProp: Coherent Hierarchical Forecasting of Building Energy Consumption Using End-to-End Deep Probabilistic Model

This repository implements **TimesProp**, a hierarchical probabilistic forecasting framework.  
It supports coherent probabilistic forecasting for two-level tree-structured time series.

## Project Structure

```
TimesProp/
│
├── train.py                # Main training script
├── timesprop.py            # Core model definition
├── metrics.py              # Evaluation metrics
├── Losses.py               # Distribution-based loss functions
├── kit.py                  # Hierarchical reconciliation methods
├── nom_tool.py             # Data scaling tools
├── timesprop_tool.py       # Period-aware feature blocks
└── data/                   # Example data files (e.g., nz21.csv, nz23.csv)
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your_username>/TimesProp.git
   cd TimesProp
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset:**
   Place CSV files in:
   ```
   /workspace/data/
   ```
   Each CSV should include a time column followed by aggregate and child nodes:
   ```
   time, total, kitchen, laundry, lighting, oven, others
   ```

---

## Usage

You can train the model using the command line:

```bash
python3 train.py \
  --path nz23.csv \
  --H 168 \
  --batch_size 64 \
  --repeat 3 \
  --lr 5e-4 \
  --p 3 \
  --dist GD \
  --period "[24, 12, 6]" \
  --part "[[2, 3, 4], [1], [1]]" \
  --MHA 1 \
  --d_model 32
```

---

## Argument Description

| Argument       | Description                                                  | Default             |
| -------------- | ------------------------------------------------------------ | ------------------- |
| `--path`       | Dataset filename (inside `/workspace/data`)                  | **required**        |
| `--H`          | Input history length (sequence length)                       | 168                 |
| `--batch_size` | Batch size                                                   | 64                  |
| `--repeat`     | Number of training runs                                      | 3                   |
| `--lr`         | Learning rate                                                | 1e-4                |
| `--p`          | Early stopping patience                                      | 3                   |
| `--dist`       | Distribution type (`D`, `GD`, `BL`, `SD`, `SSD`)             | **required**        |
| `--period`     | List of periods for multi-scale decomposition                | `[24, 12, 6]`       |
| `--part`       | Number of partitions per period                              | `[[2,3,4],[1],[1]]` |
| `--MHA`        | Multi-Head Attention module for feature interaction (0 or 1) | 1                   |
| `--d_model`    | Embedding dimension                                          | 32                  |

---

## Evaluation Metrics

| Metric      | Description                              |
| ----------- | ---------------------------------------- |
| **sMAPE**   | Symmetric Mean Absolute Percentage Error |
| **R²**      | Coefficient of determination             |
| **E_Acc≠0** | Non-Zero Disaggregation Accuracy         |
| **CRPS**    | Continuous Ranked Probability Score      |
| **MAE**     | Mean absolute error                      |

---
