import numpy as np
import torch.nn.functional as Func
from scipy.stats import dirichlet, norm, beta
import json
import math
import torch
import math
import properscoring as ps

def compute_crps_per_node(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    B, F, N, C = pred.shape
    crps_all = np.zeros((B, F, C))

    for b in range(B):
        for f in range(F):
            for c in range(C):
                crps_all[b, f, c] = ps.crps_ensemble(true[b, f, c], pred[b, f, :, c])

    crps_per_node = crps_all.mean(axis=(0, 1))
    return crps_per_node

def calculate_r_squared(predictions, truths):
    r_squared_values = []
    for i in range(predictions.shape[-1]):
        pred = predictions[:, :, i]
        true = truths[:, :, i]
        y_mean = np.mean(true)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append(r_squared)
    return np.array(r_squared_values)

def compute_mape(pred, true, eps=1e-8):
    denom = np.abs(true)
    mape = np.zeros_like(denom, dtype=np.float64)
    mask = denom > eps
    mape[mask] = np.abs(pred[mask] - true[mask]) / denom[mask]
    return mape

def calculate_normalized_quantile_loss_per_dim(pred_samples, truth, root_idx, quantiles=None):
    if quantiles is None:
        quantiles = np.arange(0.1, 1.0, 0.1)
    else:
        quantiles = np.array(quantiles)
        
    if pred_samples.ndim == 4:
        if pred_samples.shape[-1] == truth.shape[-1]:
            pred_samples = np.transpose(pred_samples, (0, 1, 3, 2))
    else:
        raise ValueError(f"pred_samples must be 4 dim")
        
    assert pred_samples.shape[:3] == truth.shape, f"Shape mismatch"
    
    preds_q = np.quantile(pred_samples, q=quantiles, axis=-1)
    diff = truth[np.newaxis, ...] - preds_q  
    q_tensor = quantiles[:, np.newaxis, np.newaxis, np.newaxis]
    loss_all_q = np.maximum(q_tensor * diff, (q_tensor - 1) * diff)
    abs_loss_sum = np.sum(loss_all_q, axis=(1, 2))
    if isinstance(root_idx, (list, np.ndarray)):
        actual_root = root_idx[0]
    else:
        actual_root = root_idx
        
    global_scale = np.sum(np.abs(truth[..., actual_root]))
    normalized_loss_per_q = abs_loss_sum / (global_scale + 1e-8)    
    quantile_loss_mean = np.mean(normalized_loss_per_q, axis=0)  
    
    return quantile_loss_mean

def calculate_mape_per_dim(merged_pred_samples_mean, merged_truth):
    assert merged_pred_samples_mean.shape == merged_truth.shape
    mape_values = compute_mape(merged_pred_samples_mean, merged_truth)
    mape_values_mean = np.mean(mape_values, axis=(0, 1))  # (10,)
    mape_show = np.mean(mape_values, axis=1)  # (batch_num, 10)
    return mape_values_mean

def calculate_E_Acc_not_zero(y_true, y_pred):
    y_true_mean = np.sum(y_true, axis=-1)  # Shape: (b, F)
    abs_errors = np.abs(y_pred - y_true)  # Shape: (b, F, C)
    mask = y_true != 0  # Shape: (b, F, C)
    masked_errors = abs_errors * mask
    numerator = np.sum(masked_errors, axis=(1, 2))  # Shape: (b,)
    denominator = 2 * np.sum(y_true_mean, axis=1)  # Shape: (b,)
    metric = 1 - (numerator / denominator)  # Shape: (b,)
    return np.mean(metric)