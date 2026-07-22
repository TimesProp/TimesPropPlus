# TimesPropPlus: Hierarchical Forecasting of Building Energy Consumption Using a Two-Stage Deep Probabilistic Model via Bayesian Learning

This repository contains the implementation of the proposed two-stage hierarchical probabilistic forecasting framework developed in this thesis.

## Project Structure

```
.
├── go_phase1_3_b.py      # Stage 1: Bayesian probabilistic forecasting
├── go_phase2_5_b4.py     # Stage 2: Hierarchical reconciliation
├── sep_n10.py            # Main forecasting model
├── B_model.py            # Bayesian neural network wrapper
├── Losses.py             # Distribution loss functions
├── metrics.py            # Evaluation metrics
├── sampling_methods.py   # Distribution sampling methods
├── data_tools.py         # Data preprocessing and dataloader
├── attack.py             # Adversarial attack methods
├── nom_tool.py           # Data normalization utilities
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- SciPy
- matplotlib
- properscoring

Install the required packages using

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1

Train the Bayesian probabilistic forecasting model:

```bash
python go_phase1_3_b.py
```

### Stage 2

Train the reconciliation layer using the outputs generated in Stage 1:

```bash
python go_phase2_5_b4.py
```

## Notes

- Stage 1 generates probabilistic forecasts using Bayesian learning.
- Stage 2 learns a reconciliation layer to produce coherent hierarchical forecasts.
- Dataset paths and training parameters can be modified through the command-line arguments in each training script.
