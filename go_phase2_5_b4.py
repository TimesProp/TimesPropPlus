import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from types import SimpleNamespace
import os
import argparse
import ast
import copy
import sys
import metrics
import data_tools
from sep_n10 import Model as Model10
from B_model2 import B_SimpleNN

class DRLOfflineDataset(Dataset):
    def __init__(self, preds_path, trues_path):
        self.preds = np.load(preds_path).astype(np.float32)
        self.trues = np.load(trues_path).astype(np.float32)
        self.B_total, self.F_len, self.N_samples, self.Num_Nodes = self.preds.shape
        self.flat_preds = self.preds.reshape(-1, self.N_samples, self.Num_Nodes)
        self.flat_trues = self.trues.reshape(-1, self.Num_Nodes)

    def __len__(self):
        return self.flat_preds.shape[0]

    def __getitem__(self, idx):
        return self.flat_preds[idx], self.flat_trues[idx]

def energy_score_loss(reconciled_samples, true_y, beta=1.0):
    if reconciled_samples.dim() == 4:
        B, F, N_samples, Num_Nodes = reconciled_samples.shape
        reconciled_samples = reconciled_samples.reshape(-1, N_samples, Num_Nodes)
        true_y = true_y.reshape(-1, Num_Nodes)

    true_y = true_y.unsqueeze(1) 
    term1 = torch.norm(reconciled_samples - true_y, p=2, dim=-1).pow(beta).mean(dim=1)
    
    samples1 = reconciled_samples.unsqueeze(2) 
    samples2 = reconciled_samples.unsqueeze(1) 
    pairwise_diff = torch.norm(samples1 - samples2, p=2, dim=-1).pow(beta)
    term2 = 0.5 * pairwise_diff.mean(dim=(1, 2))
    return (term1 - term2).mean()

class DRLLayerModel(nn.Module):
    def __init__(self, num_nodes, S_matrix_np, hidden_dim=32):
        super(DRLLayerModel, self).__init__()
        self.num_nodes = num_nodes
        
        self.q_net = nn.Sequential(
            nn.BatchNorm1d(num_nodes),
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )
        
        all_parents = set([rule[0] for rule in S_matrix_np])
        all_nodes = set(range(num_nodes))
        leaf_nodes = sorted(list(all_nodes - all_parents)) 
        num_leaves = len(leaf_nodes)
        
        S = np.zeros((num_nodes, num_leaves), dtype=np.float32)
        for i, leaf in enumerate(leaf_nodes):
            S[leaf, i] = 1.0
            
        for _ in range(5):
            for rule in S_matrix_np:
                p_idx, c_idices = rule[0], rule[1]
                S[p_idx, :] = np.sum([S[c, :] for c in c_idices], axis=0)
                
        self.register_buffer('S', torch.from_numpy(S))

    def forward(self, batch_samples):
        device = batch_samples.device
        
        orig_shape = batch_samples.shape  
        Num_Nodes = orig_shape[-1]
        N_samples = orig_shape[-2]
        flat_samples = batch_samples.view(-1, N_samples, Num_Nodes)
        N = flat_samples.shape[0] 
        
        mean_preds = flat_samples.mean(dim=1)
        
        raw_q = self.q_net(mean_preds)
        
        q_vals = torch.nn.functional.softplus(raw_q)
        q_vals = torch.clamp(q_vals, min=1e-3, max=1e3) # (N, Num_Nodes)
        
        QS = q_vals.unsqueeze(-1) * self.S.unsqueeze(0)  
        S_T = self.S.t().unsqueeze(0).expand(N, -1, -1)
        denom = torch.bmm(S_T, QS)
        
        denom_double = denom.to(torch.float64)
        QS_T_double = QS.transpose(1, 2).to(torch.float64)
        jitter = 1e-5 * torch.eye(self.S.shape[1], dtype=torch.float64, device=device).unsqueeze(0)
        
        result = torch.linalg.lstsq(denom_double + jitter, QS_T_double)
        X = result.solution.to(torch.float32)
        
        S_batch = self.S.unsqueeze(0).expand(N, -1, -1)
        P = torch.bmm(S_batch, X)
        
        if torch.isnan(P).any():
            dummy_grad = 0.0 * raw_q.sum()
            safe_samples = flat_samples + dummy_grad
            return safe_samples.view(orig_shape)
        
        reconciled = torch.bmm(flat_samples, P.transpose(1, 2))
        return reconciled.view(orig_shape)

class AttackModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x, x_mark=None, wrap_3=None, wrap_4=None):
        families, samples, prob = self.model(x, x_mark, wrap_3, wrap_4)
        return families, samples

def attack_two_stage(model_s1, x, x_mark, y, att_method, epsilon, mode="untarget", iters=5, alpha=1/255, scale_mean=None, scale_std=None):
    if att_method == 'None' or att_method is None:
        return x
        
    x_adv = x.clone().detach().requires_grad_(True)
    loop_iters = iters if att_method == 'PGD' else 1
    
    attack_wrapper = AttackModelWrapper(model_s1)
    
    if scale_mean is not None and scale_std is not None:
        y_norm = (y - scale_mean) / scale_std
    else:
        y_norm = y 

    for i in range(loop_iters):
        model_s1.train()
        model_s1.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        
        _, samples_norm = attack_wrapper(x_adv, x_mark, None, None)
        point_pred_norm = samples_norm.mean(dim=2)  # (B, F, N)
        
        loss = nn.functional.mse_loss(point_pred_norm, y_norm)
        loss.backward()
        
        with torch.no_grad():
            if x_adv.grad is None:
                break
            grad = x_adv.grad.sign()
            if mode == "untarget":
                x_adv = x_adv + alpha * grad  
            else:
                x_adv = x_adv - alpha * grad
            
            eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = x + eta 
            
        x_adv = x_adv.detach().requires_grad_(True)
        
    model_s1.zero_grad() 
    model_s1.eval()
    return x_adv.detach()

def formular(input_res, levels_idx):
    levels_input_res = [[input_res[i] for i in level] for level in levels_idx]
    levels_input_res_mean = [np.nanmean(np.stack(level, axis=0), axis=0) for level in levels_input_res]
    return np.stack(levels_input_res_mean, axis=0) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 2: Reconciliation Training and Hierarchical Metrics Log")
    parser.add_argument("--DRL_data_dir", type=str, default="/workspace/DRL_dataset", help="Data directory")
    parser.add_argument("--prefix_base", type=str, required=True, help="File prefix without run index")
    parser.add_argument("--hier", type=str, required=True, help="Path to structure adjacency matrix CSV")
    parser.add_argument("--path", type=str, required=True, help="Original data CSV path")
    parser.add_argument("--repeat", type=int, default=1, help="Number of run repeats")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--note", type=str, default="Stage2", help="Note")
    parser.add_argument("--att", type=str, nargs='+', default=["None", "FGSM", "PGD"], help="Attack methods list")
    parser.add_argument("--epsilon", type=float, nargs='+', default=[0.005, 0.01, 0.015, 0.02, 0.025], help="List of perturbation strengths")       
    parser.add_argument("--mode", type=str, default="untarget", help="Attack mode")
    parser.add_argument("--H", type=int, default=168, help="History")
    parser.add_argument("--F", type=int, default=24, help="Horizon")
    parser.add_argument("--stage1_weights_dir", type=str, default="/workspace/timesprop-b/weights", help="Weight cache dir")
    parser.add_argument("--period", type=ast.literal_eval, default=[24], help="Period List")
    parser.add_argument("--part", type=ast.literal_eval, default=[[2]], help="Part List")
    parser.add_argument("--dist", type=str, default="GD", help="Dist type")
    parser.add_argument("--t", type=int, default=10, help="Period Type")
    parser.add_argument("--d_model", type=int, default=64, help="d_model")
    parser.add_argument("--MHA", type=int, default=0, help="MHA flag")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_path = os.path.join("/workspace/data", args.path)
    full_hier_path = os.path.join("/workspace/data", args.hier)
    df = pd.read_csv(full_path)
    families, node2idx = data_tools.load_tree_families(full_hier_path, df)
    num_nodes = len(node2idx)
    hierarchy_rules = [[fam["parent"], fam["children"]] for fam in families]
    
    method_names = ["Unreconciled", "Reconciled"]
    levels_idx = data_tools.build_levels_from_families(families, len(node2idx))

    data, x_mark = data_tools.load_dataset(df)
    folds = data_tools.temporal_fold_split2(data, x_mark, fold=1, p_train=0.7, p_val=0.1)
    _, _, test_sets = data_tools.build_datasets_from_folds(folds, seq_len=args.H, pred_len=args.F)
    _, _, test_loader = data_tools.build_loaders(_, _, test_sets, batch_size=args.batch_size)

    print(f'Starting reconciliation evaluation across methods {args.att} and epsilons {args.epsilon}')

    print("\n" + "="*70)
    print("="*70)
    DRL_pretrained_weights = {}

    for repeat in range(args.repeat):
        print(f"\n--> Pre-training Matrix for Source: {args.prefix_base}_run{repeat}_bayes")
        prefix = f"{args.prefix_base}_run{repeat}_bayes"

        train_preds_path = os.path.join(args.DRL_data_dir, f"{prefix}_DRL_train_preds.npy")
        train_trues_path = os.path.join(args.DRL_data_dir, f"{prefix}_DRL_train_trues.npy")
        val_preds_path   = os.path.join(args.DRL_data_dir, f"{prefix}_DRL_val_preds.npy")
        val_trues_path   = os.path.join(args.DRL_data_dir, f"{prefix}_DRL_val_trues.npy")

        if not os.path.exists(train_preds_path):
            print(f"Missing: {train_preds_path}, Skipping")
            DRL_pretrained_weights[repeat] = None
            continue

        train_dataset = DRLOfflineDataset(train_preds_path, train_trues_path)
        val_dataset   = DRLOfflineDataset(val_preds_path, val_trues_path)
        train_loader_DRL = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader_DRL   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        DRL_model = DRLLayerModel(num_nodes=num_nodes, S_matrix_np=hierarchy_rules).to(device)
        optimizer = optim.Adam(DRL_model.parameters(), lr=args.lr)
        best_val_loss = float('inf')
        best_Qr_weights = None

        for epoch in range(args.epochs):
            print(f"{epoch} / {args.epochs}")
            DRL_model.train()
            for preds, trues in train_loader_DRL:
                optimizer.zero_grad()
                preds, trues = preds.to(device), trues.to(device)
                reconciled = DRL_model(preds)
                loss = energy_score_loss(reconciled, trues)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DRL_model.parameters(), max_norm=1.0)
                optimizer.step()

            DRL_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for preds, trues in val_loader_DRL:
                    preds, trues = preds.to(device), trues.to(device)
                    reconciled = DRL_model(preds)
                    val_loss = energy_score_loss(reconciled, trues)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader_DRL)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_Qr_weights = copy.deepcopy(DRL_model.state_dict())

        for k, v in best_Qr_weights.items():
            best_Qr_weights[k] = v.cpu()
        DRL_pretrained_weights[repeat] = best_Qr_weights
        print(f"   [Run {repeat}] Pre-training completed. Best Val Loss: {best_val_loss:.4f}")
        
        del DRL_model
        torch.cuda.empty_cache()

    for current_att in args.att:
        for current_eps in args.epsilon:
            if current_att == "None" and current_eps != args.epsilon[0]:
                continue
                
            print(f"\n==========================================================================")
            print(f"Running Global Bayesian Evaluation -> Attack: [{current_att}] | Epsilon: [{current_eps}]")
            print(f"==========================================================================")

            results = {
                name: {"smape": [], "r2": [], "eacc": [], "crps": [], "smape_children": [], "level_smape":[],
                        "level_mape": [], "level_r2":[], "level_crps":[],
                      "eacc_leaf": [], "eacc_mid": [], "level_ql": []}
                for name in method_names
            }

            for repeat in range(args.repeat):
                if DRL_pretrained_weights.get(repeat) is None:
                    continue

                prefix = f"{args.prefix_base}_run{repeat}_bayes"
                print(f"\n--> [Att: {current_att} | Eps: {current_eps}] | Evaluating Run: {prefix}")

                DRL_model = DRLLayerModel(num_nodes=num_nodes, S_matrix_np=hierarchy_rules).to(device)
                DRL_model.load_state_dict(DRL_pretrained_weights[repeat])
                DRL_model.to(device)
                DRL_model.eval()

                print(f">> Loading Stage 1 Base Model and wrapping into Bayesian Container...")
                configs = SimpleNamespace(
                    task_name='long_term_forecast', seq_len=args.H, pred_len=args.F, d_model=args.d_model,
                    top_k=5, num_nodes=num_nodes, num_kernels=6, embed='timeF', freq='h', dropout=0.1,
                    num_class=0, dist_type=args.dist, t_type=args.t, period_list=args.period,
                    part_list=args.part, mha=args.MHA, families=families, sample_size=100  
                )
                
                base_model_s1 = Model10(configs).to(device)
                else: raise ValueError(f"Unsupported model string: {args.m}")
                    
                model_s1 = B_SimpleNN(base_model_s1, prior_std=1.0).to(device)
                stage1_pth = os.path.join(args.stage1_weights_dir, args.path.split('.')[0] + "_saved_bayes.pth")
                model_s1.load_state_dict(torch.load(stage1_pth, map_location=device))
                model_s1.eval()
                
                all_unrecon_samples = []
                all_recon_samples = []
                all_y_trues = []

                print(f">> Running dynamic inference forward loop under [{current_att}] attack...")

                for batch_idx, (x, y, x_mark) in enumerate(test_loader):
                    x, y = x.to(device), y.to(device)
                    if x_mark is not None: x_mark = x_mark.to(device)
                    
                    scale_mean = x.mean(dim=1, keepdim=True)
                    scale_std = x.std(dim=1, keepdim=True) + 1e-5
                    x_input = (x - scale_mean) / scale_std 

                    x_adv = attack_two_stage(
                        model_s1, x_input, x_mark, y, current_att, current_eps, 
                        mode=args.mode, scale_mean=scale_mean, scale_std=scale_std
                    )

                    model_s1.eval()
                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            model_output = model_s1.bayes_forward(x_adv, x_mark, None, None, K=10)
                        samples_norm = model_output[1]  
                        
                        scale_mean_b = scale_mean.unsqueeze(2)
                        scale_std_b = scale_std.unsqueeze(2)
                        samples_real = samples_norm * scale_std_b + scale_mean_b
                        
                        reconciled_samples = DRL_model(samples_real)

                    all_unrecon_samples.append(samples_real.cpu().numpy())
                    all_recon_samples.append(reconciled_samples.cpu().numpy())
                    all_y_trues.append(y.cpu().numpy())

                orig_test_samples = np.concatenate(all_unrecon_samples, axis=0)
                recon_test_samples = np.concatenate(all_recon_samples, axis=0)
                y_true = np.concatenate(all_y_trues, axis=0)

                print(
                    f"DEBUG -> Ground Truth Mean: {y_true.mean():.4f} | "
                    f"Bayesian Prediction Mean: {orig_test_samples.mean():.4f} | "
                    f"Reconciled Prediction Mean: {recon_test_samples.mean():.4f}"
                )

                samples_dict = {
                    "Unreconciled": orig_test_samples,   
                    "Reconciled": recon_test_samples  
                }
                preds_dict = {
                    "Unreconciled": orig_test_samples.mean(axis=2),  
                    "Reconciled": recon_test_samples.mean(axis=2) 
                }

                root_indices = levels_idx[0]
                mid_indices  = levels_idx[1]
                leaf_indices = levels_idx[2]

                for method in method_names:
                    pred = preds_dict[method] 
                    outs = samples_dict[method] 

                    print(f">> Metric Check | {prefix} | Method: {method} | Pred: {pred.shape}")
                    
                    mape = metrics.calculate_mape_per_dim(pred, y_true)            
                    levels_mape_mean = formular(mape, levels_idx)

                    r2 = metrics.calculate_r_squared(pred, y_true)
                    r2 = np.where(np.isinf(r2), np.nan, r2) 
                    levels_r2_mean = formular(r2, levels_idx)
                    
                    crps_all_nodes = metrics.compute_crps_per_node(outs, y_true)
                    levels_crps_mean = formular(crps_all_nodes, levels_idx)

                    eacc = metrics.calculate_E_Acc_not_zero(y_true[..., 1:], pred[..., 1:])
                    eacc_leaf = metrics.calculate_E_Acc_not_zero(y_true[..., leaf_indices], pred[..., leaf_indices])
                    eacc_mid = metrics.calculate_E_Acc_not_zero(y_true[..., mid_indices], pred[..., mid_indices])

                    ql_per_node = metrics.calculate_normalized_quantile_loss_per_dim(outs, y_true, root_indices[0])
                    levels_ql_mean = formular(ql_per_node, levels_idx)
                    
                    results[method]["level_mape"].append(levels_mape_mean)
                    results[method]["level_r2"].append(levels_r2_mean)
                    results[method]["level_crps"].append(levels_crps_mean)
                    results[method]["eacc"].append(eacc)
                    results[method]["eacc_leaf"].append(eacc_leaf)
                    results[method]["eacc_mid"].append(eacc_mid)
                    results[method]["crps"].append(crps_all_nodes)
                    results[method]["level_ql"].append(levels_ql_mean)

                print(f"Completed run mapping for loop index {repeat}")
                del DRL_model, model_s1
                torch.cuda.empty_cache()

            print(f"\n======== FINAL BAYESIAN RESULTS FOR ATTACK: {current_att} | EPS: {current_eps} ========")
            formatted_note = f"{args.note}_Bayes_{current_att}_Eps_{current_eps:.4f}"
            print(args.path)
            for method in method_names:            

                print("#############")
                level_mape = np.array(results[method]["level_mape"])
                level_mape_mean = level_mape.mean(axis=0) 
                level_mape_std = level_mape.std(axis=0) 
                content = '\t'.join(f"{level_mape_mean[idx]:.4f}±{level_mape_std[idx]:.4f}" for idx in range(len(level_mape_mean)))
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"mape"+"\t"+content
                print(line)

                print("###")
                level_r2 = np.array(results[method]["level_r2"])
                level_r2_mean = level_r2.mean(axis=0) 
                level_r2_std = level_r2.std(axis=0) 
                content = '\t'.join(f"{level_r2_mean[idx]:.4f}±{level_r2_std[idx]:.4f}" for idx in range(len(level_r2_mean)))
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"r2"+"\t"+content
                print(line)
                
                print("###")
                level_crps = np.array(results[method]["level_crps"])
                level_crps_mean = level_crps.mean(axis=0) 
                level_crps_std = level_crps.std(axis=0) 
                content = '\t'.join(f"{level_crps_mean[idx]:.4f}±{level_crps_std[idx]:.4f}" for idx in range(len(level_crps_mean)))
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"crps"+"\t"+content
                print(line)
                
                print("###")
                level_res = np.array(results[method]["eacc"])
                content = f"{level_res.mean():.4f}" 
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"eacc"+"\t"+content
                print(line)
                
                print("###")
                level_res = np.array(results[method]["eacc_leaf"])
                level_res_mean = level_res.mean(axis=0) 
                level_res_std = level_res.std(axis=0) 
                content = f"{level_res_mean:.4f}±{level_res_std:.4f}"
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"eacc_leaf"+"\t"+content
                print(line)
                
                print("###")
                level_res = np.array(results[method]["eacc_mid"])
                level_res_mean = level_res.mean(axis=0) 
                level_res_std = level_res.std(axis=0) 
                content = f"{level_res_mean:.4f}±{level_res_std:.4f}"
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"eacc_mid"+"\t"+content
                print(line)
                
                print("###")
                level_res = np.array(results[method]["level_ql"])
                level_res_mean = level_res.mean(axis=0) 
                level_res_std = level_res.std(axis=0) 
                content = '\t'.join(f"{level_res_mean[idx]:.4f}±{level_res_std[idx]:.4f}" for idx in range(len(level_res_mean)))
                line = args.path+"\t"+formatted_note+"\t"+f"{method}"+"\t"+"NQL"+"\t"+content
                print(line)
                print("#############")

            print("")

    sys.exit(0)