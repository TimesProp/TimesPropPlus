import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from types import SimpleNamespace

from sep_n10 import Model as Model10
from B_model import B_SimpleNN
import matplotlib.pyplot as plt
import copy
import os
import argparse
import torch.nn.functional as F
import Losses
import time
from torch.optim.lr_scheduler import LambdaLR
import ast
import attack
import data_tools
import sys
import gc
from nom_tool import GlobalMinMaxScaler
best_model_path = '/workspace/timesprop-b/saved_model_sep77ak_bayes.pth'
warmup_epochs = 0
FOLD = 1
LOSS_CONFIGS = SimpleNamespace(
    model_name = '',
    dist_type = ''
)
MODEL_REGISTRATION = {
    "D" : {"LOSS_FUNC" : Losses.dirichlet_loss},
    "GD" : {"LOSS_FUNC" : Losses.gdd_loss},
    "BL" : {"LOSS_FUNC" : Losses.beta_liouville_loss},
    "SD" : {"LOSS_FUNC" : Losses.scaled_dirichlet_loss},
    "SSD" : {"LOSS_FUNC" : Losses.shifted_scaled_dirichlet_loss}
}

def gaussian_nll(prob, target):
    mu = prob[..., 0].clone()          
    sigma = prob[..., 1].clone()       
    sigma_safe = torch.clamp(sigma, min=1e-5, max=1e2)  

    nll_matrix = 0.5 * torch.log(2 * torch.pi * sigma_safe**2) + ((target - mu)**2) / (2 * sigma_safe**2)
    node_scales = torch.mean(torch.abs(target), dim=(0, 1), keepdim=True) + 1.0  
    balanced_nll = nll_matrix / node_scales 
    return balanced_nll.mean()

def calculate_loss(family, res, y, kl, beta, scale_mean=None, scale_std=None):
    family_output, samples, prob = res
    eps = 1e-6
    
    if scale_mean is not None and scale_std is not None:
        y_target = (y - scale_mean) / scale_std
    else:
        y_target = y

    dist_loss_func = MODEL_REGISTRATION[LOSS_CONFIGS.dist_type]["LOSS_FUNC"]
    y_tensors = []
    for fam in family:
        p = fam["parent"]
        c = fam["children"]
        idx = torch.tensor([p] + c, device=samples.device, dtype=torch.long)
        fam_y = y_target.index_select(dim=2, index=idx)
        y_tensors.append(fam_y)

    loss_list = []
    loss_list_mu = []
    loss_list_dist = []
    for idx, y_tensor in enumerate(y_tensors):
        if 'family_mu' in family_output[idx]:
            out = family_output[idx]['family_mu'] 
            loss_list_mu.append(nn.functional.mse_loss(out, y_tensor))
        if 'family_dist' in family_output[idx]:
            part = y_tensor[..., 1:]
            if part.size(-1) > 1:
                part_clamped = torch.clamp(part, min=0.0)
                sums = part_clamped.sum(dim=-1, keepdim=True)
                normalized = part_clamped / (sums + eps) 
                out = family_output[idx]['family_dist'] 
                loss_list_dist.append(dist_loss_func(normalized, out))
            
    loss_mu =  torch.stack(loss_list_mu).sum()
    loss_dist =  torch.stack(loss_list_dist).sum()
    loss2 = gaussian_nll(prob, y_target)
    loss3 = nn.functional.mse_loss(samples.mean(dim=2), y_target)
    
    loss_dist = loss_dist * 0.1
    loss2 = loss2
    loss3 = loss3 * 1
    task_loss = loss_mu + loss_dist + loss2 + loss3
    kl_loss = kl * beta
    
    loss_sum = task_loss + kl_loss
    return loss_sum, [loss_mu, loss_dist, loss2, loss3, kl_loss]

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, families, epochs=50, patience=3, lp=[1, 0, 0], dist_type='SSD', beta_bnn=5e-5):
    best_model = None
    best_loss = float('inf')
    patience_counter = 0

    scaler_amp = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loss_comps = None
        
        for batch_idx, (x, y, x_mark) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_mark = x_mark.to(device, non_blocking=True) if x_mark is not None else None

            scale_mean = x.mean(dim=1, keepdim=True)         
            scale_std = x.std(dim=1, keepdim=True) + 1e-5    
            x_norm = (x - scale_mean) / scale_std

            with torch.amp.autocast('cuda'):
                model_output = model(x_norm, x_mark, None, None)
                kl = model.kl_divergence() / x.shape[0]
                loss, loss_comps = calculate_loss(families, model_output, y, kl, beta_bnn, scale_mean, scale_std)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item()
            
            comps_val = [c.item() if isinstance(c, torch.Tensor) else float(c) for c in loss_comps]
            if train_loss_comps is None:
                train_loss_comps = comps_val
            else:
                train_loss_comps = [sum_c + v for sum_c, v in zip(train_loss_comps, comps_val)]
            
        scheduler.step()
        
        train_loss = total_loss / len(train_loader)
        train_loss_comps = [c / len(train_loader) for c in train_loss_comps]

        model.eval()
        val_loss = 0
        val_loss_comps = None
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x, y, x_mark) in enumerate(val_loader):
                try:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    x_mark = x_mark.to(device, non_blocking=True) if x_mark is not None else None
                    
                    scale_mean = x.mean(dim=1, keepdim=True)
                    scale_std = x.std(dim=1, keepdim=True) + 1e-5
                    x_norm = (x - scale_mean) / scale_std

                    with torch.amp.autocast('cuda'):
                        model_output = model(x_norm, x_mark, None, None)
                        kl = model.kl_divergence() / x.shape[0]
                        loss, loss_comps = calculate_loss(families, model_output, y, kl, beta_bnn, scale_mean, scale_std)
                    
                    val_loss += loss.item()
                    
                    comps_val = [c.item() if isinstance(c, torch.Tensor) else float(c) for c in loss_comps]
                    if val_loss_comps is None:
                        val_loss_comps = comps_val
                    else:
                        val_loss_comps = [sum_c + v for sum_c, v in zip(val_loss_comps, comps_val)]
                    
                    val_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        continue
                    else: 
                        raise e
        
        if val_batches > 0:
            val_loss /= val_batches
            val_loss_comps = [c / val_batches for c in val_loss_comps]
        else:
            val_loss = float('inf')
            val_loss_comps = []

        train_comps_str = "[" + ", ".join([f"{c:.4f}" for c in train_loss_comps]) + "]" if train_loss_comps else "[]"
        val_comps_str = "[" + ", ".join([f"{c:.4f}" for c in val_loss_comps]) + "]" if val_loss_comps else "[]"

        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {train_loss:.6f} {train_comps_str} | Val Loss: {val_loss:.6f} {val_comps_str}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    torch.save(best_model, best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model

def extract_dataset_for_stage2(model, loader, device, is_val=True):
    model.eval()
    all_samples_real = []
    all_trues_real = []
    
    print(f">> Pre-extracting samples using Bayesian Monte Carlo Sampling from {'Validation Loader' if is_val else 'Test Loader'}...")
    with torch.no_grad():
        for batch_idx, (x, y, x_mark) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_mark = x_mark.to(device, non_blocking=True) if x_mark is not None else None
            
            scale_mean = x.mean(dim=1, keepdim=True)          
            scale_std = x.std(dim=1, keepdim=True) + 1e-5     
            x_norm = (x - scale_mean) / scale_std

            with torch.amp.autocast('cuda'):
                model_output = model.bayes_forward(x_norm, x_mark, None, None, K=10)
            samples_norm = model_output[1]  # (B, F, num_samples, N)
            
            scale_mean_b = scale_mean.unsqueeze(2)
            scale_std_b = scale_std.unsqueeze(2)
            
            samples_real = samples_norm * scale_std_b + scale_mean_b
            
            all_samples_real.append(samples_real.cpu())
            all_trues_real.append(y.cpu())
            
    all_samples_real = torch.cat(all_samples_real, dim=0).numpy()
    all_trues_real = torch.cat(all_trues_real, dim=0).numpy()
    return all_samples_real, all_trues_real


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 1: Bayesian Predictive Model Training")
    parser.add_argument("--path", type=str, required=True, help="File path relative to /workspace/data")
    parser.add_argument("--H", type=int, default=168, help="History")
    parser.add_argument("--F", type=int, default=24, help="Predict")
    parser.add_argument("--repeat", type=int, default=1, help="Stage 1 repeats")
    parser.add_argument("--load", action="store_true", help="Load weights")
    parser.add_argument("--savetest", action="store_true", help="Save arrays")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_sample", type=int, default=1, help="Using sample")
    parser.add_argument("--lp", type=int, nargs='+', default=[1, 0, 0], help="Loss prop")
    parser.add_argument("--p", type=int, default=5, help="Patience")
    parser.add_argument("--dist", type=str, required=True, help="Dist type")
    parser.add_argument("--t", type=int, default=8, help="Period Type")
    parser.add_argument("--MHA", type=int, default=0, help="MHA flag")
    parser.add_argument("--d_model", type=int, default=32, help="d_model")
    parser.add_argument("--period", type=ast.literal_eval, help="Period List")
    parser.add_argument("--part", type=ast.literal_eval, help="Part List")
    parser.add_argument("--hier", type=str, required=True, help="Structure")
    parser.add_argument("--note", type=str, default="NaN", help="Note")
    parser.add_argument("--DRL_save_dir", type=str, default="/workspace/DRL_dataset", help="Path to save stage2 npy files")
    parser.add_argument("--beta_bnn", type=float, default=5e-5, help="KL loss trade-off coefficient")

    args = parser.parse_args()
    full_path = os.path.join("/workspace/data", args.path)
    model_save_path_folder = "/workspace/timesprop-b/weights"
    save_path = os.path.join(model_save_path_folder, args.path.split('.')[0]+"_saved_bayes.pth")
    if not os.path.exists(model_save_path_folder):
        os.makedirs(model_save_path_folder)
    full_hier_path = os.path.join("/workspace/data", args.hier)
    best_model_path = save_path

    df = pd.read_csv(full_path)
    families, node2idx = data_tools.load_tree_families(full_hier_path, df)
    num_nodes = len(node2idx)

    print(f'Starting Stage 1 Bayesian Predictive Training (fold={FOLD})...')
    
    for repeat in range(args.repeat):
        print(f"======== Training Repeat {repeat + 1}/{args.repeat} ========")

        configs = SimpleNamespace(
            task_name='long_term_forecast', seq_len=args.H, pred_len=args.F, d_model=args.d_model, 
            top_k=5, num_nodes=num_nodes, num_kernels=6, embed='timeF', freq='h', dropout=0.1,
            num_class=0, dist_type=args.dist, t_type=args.t, period_list=args.period,
            part_list=args.part, mha=args.MHA, families=families, sample_size=1000  
        )
        LOSS_CONFIGS.dist_type = args.dist

        data, x_mark = data_tools.load_dataset(df)
        folds = data_tools.temporal_fold_split2(data, x_mark, fold=FOLD, p_train=0.7, p_val=0.1)
        
        train_sets, val_sets, test_sets = data_tools.build_datasets_from_folds(folds, seq_len=args.H, pred_len=args.F)
        train_loader, val_loader, test_loader = data_tools.build_loaders(train_sets, val_sets, test_sets, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        base_model = Model10(configs).to(device)
        model = B_SimpleNN(base_model, prior_std=1.0).to(device)
        print(f"Bayesian Wrapper for Model initialized.")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, 1)
        criterion = nn.MSELoss()

        if args.load:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Bayesian Model weights loaded from cache.")
        else:
            model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, 
                                families, epochs=50, patience=args.p, lp=args.lp, dist_type=args.dist, beta_bnn=args.beta_bnn)
        
        if not os.path.exists(args.DRL_save_dir):
            os.makedirs(args.DRL_save_dir)
            
        val_samples, val_trues = extract_dataset_for_stage2(model, val_loader, device, is_val=True)
        
        split_idx = int(val_samples.shape[0] * 0.8)
        DRL_train_preds = val_samples[:split_idx]
        DRL_train_trues = val_trues[:split_idx]
        DRL_val_preds = val_samples[split_idx:]
        DRL_val_trues = val_trues[split_idx:]
        
        DRL_test_preds, DRL_test_trues = extract_dataset_for_stage2(model, test_loader, device, is_val=False)
        
        prefix = f"{args.path.split('.')[0]}_run{repeat}_bayes"
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_train_preds.npy"), DRL_train_preds)
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_train_trues.npy"), DRL_train_trues)
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_val_preds.npy"), DRL_val_preds)
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_val_trues.npy"), DRL_val_trues)
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_test_preds.npy"), DRL_test_preds)
        np.save(os.path.join(args.DRL_save_dir, f"{prefix}_DRL_test_trues.npy"), DRL_test_trues)
        
        print(f"\n>>> [SUCCESS] Stage 1 exports completed for Bayesian runtime index: {repeat}.")

        del model, optimizer, train_loader, val_loader, test_loader, val_samples, val_trues
        gc.collect()
        torch.cuda.empty_cache()

    print("======== ALL STAGE-1 OPERATIONS COMPLETED ========")
    sys.exit(0)