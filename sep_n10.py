import torch
import torch.nn as nn
import torch.nn.functional as F
import sampling_methods as sampling
import sep4_kit
from sep4_kit import PeriodAwareConvBlock


DISTRIBUTION_REGISTRY = {
    "D": {"sampler": sampling.sample_dirichlet,
        "res_dim_fn": lambda C: C},
    "GD": {"sampler": sampling.sample_gd,
        "res_dim_fn": lambda C: 2 * (C - 1)},
    "BL": {"sampler": sampling.sample_bl,
        "res_dim_fn": lambda C: C + 1},
    "SD": {"sampler": sampling.scaled_dirichlet_sampling,
        "res_dim_fn": lambda C: 2 * C},
    "SSD": {"sampler": sampling.sample_ssd_batch,
        "res_dim_fn": lambda C: 2 * C + 1}
}

def restore_from_families(
    families,
    family_tensors,
    enc_x,
    mode="child"
):
    assert mode in ["child", "parent"]

    def _child_priority():
        out = enc_x.clone()
        for fam, fam_x in zip(families, family_tensors):
            p = fam["parent"]
            out[..., p] = fam_x[..., 0]
        for fam, fam_x in zip(families, family_tensors):
            c = fam["children"]
            for i, ci in enumerate(c):
                out[..., ci] = fam_x[..., i + 1]
        return out

    def _parent_priority():
        out = enc_x.clone()
        for fam, fam_x in zip(families, family_tensors):
            c = fam["children"]
            for i, ci in enumerate(c):
                # out[:, :, :, ci] = fam_x[:, :, :, i + 1]
                out[..., ci] = fam_x[..., i + 1]
        for fam, fam_x in zip(families, family_tensors):
            p = fam["parent"]
            out[..., p] = fam_x[..., 0]
        return out

    if mode == "child":
        return _child_priority()
    if mode == "parent":
        return _parent_priority()

class PeriodBlock(nn.Module):
    def __init__(self, configs):
        super(PeriodBlock, self).__init__()
        self.mha = configs.mha
        self.t_type = configs.t_type
        self.period_list = configs.period_list
        self.part_list = configs.part_list
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.num_nodes
        self.period_conv2 = nn.ModuleList([
            PeriodAwareConvBlock(self.d_model, self.d_model)
            for _ in range(len(self.period_list))
        ])
        
        
    def forward(self, x, x_mark_enc):
        B, T, N = x.size()
        influence = sep4_kit.compute_period_influence(x, self.period_list) # shape (B, len(period_list))
        res = []
        for idx, period in enumerate(self.period_list):
            segments, pad_left_lens, pad_right_lens = sep4_kit.process_signal_to_2d_batch(x, x_mark_enc,
                                                                                          period, self.pred_len, True, self.t_type)

            segments = segments.permute(0, 3, 1, 2).contiguous()  # (B, N, P_num, Period)
            out1 = self.period_conv2[idx](segments)  # (B, N, P_num, Period)
            out = out1
            P_num = out.shape[2]
            out = out.permute(0, 2, 3, 1).contiguous()  # → (B, P_num, Period, N)        
            out = out.view(B, -1, out.shape[-1])  # → (B, T_ext=P_num*Period, N)
            segments = []
            for b in range(B):
                left = pad_left_lens[b]
                right = pad_right_lens[b]
                right_end = out.shape[1] - right
                if left==0 and right==0:
                    left=24
                segments.append(out[b, left:right_end, :])
            segments = torch.stack(segments, dim=0) # (B, seq_len+pred_len, N)
            res.append(segments)
        res = torch.stack(res, dim=-1) # (B, seq_len+pred_len, N, k)
        influence = influence.unsqueeze(1).unsqueeze(1)
        res = res * influence
        res = res.sum(dim=-1) # (B, seq_len+pred_len, N)
        res = res[:, :x.shape[1], :] # (B, seq_len, N)
        return res

class LSTMForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,   # C+1
        hidden_dim: int,
        num_layers: int,
        pred_len: int,    # F
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x):
        B = x.size(0)
        out, _ = self.lstm(x)          # (B, H, hidden)
        h_last = out[:, -1, :]         # (B, hidden)
        y = self.head(h_last)          # (B, F*(C+1))
        y = y.view(B, self.pred_len, self.input_dim)
        return y

class MultiHierarchyFusion(nn.Module):
    def __init__(self, N, K):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(N, K))

    def forward(self, Y_list):
        Y_stack = torch.stack(Y_list, dim=-1)
        alpha = F.softmax(self.alpha_logits, dim=-1)
        alpha = alpha.view(1, 1, *alpha.shape)
        Y = (alpha * Y_stack).sum(dim=-1)

        return Y

class GaussianHead(nn.Module):
    def __init__(self, hidden_dim, pred_len):
        super().__init__()
        self.pred_len = pred_len

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * pred_len)
        )

    def forward(self, x, B, N):
        out = self.mlp(x)  # (B*N, 2*pred_len)
        out = out.view(B, N, 2, self.pred_len)
        out = out.permute(0, 3, 1, 2)  # (B, pred_len, N, 2)
        mean = out[..., 0]
        sigma = out[..., 1]
        sigma = torch.nn.functional.softplus(sigma) + 1e-6
        return mean, sigma

class LSTMnew(nn.Module):
    def __init__(self, hidden_dim, seq_len, pred_len, N):
        super().__init__()
        self.seq_len = seq_len
        self.N = N

        self.lstms = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.ghead = GaussianHead(hidden_dim, pred_len)

    def forward(self, x):
        # (B, H, N) -> (B*N, H, 1)
        x_node = x.permute(0, 2, 1).reshape(-1, self.seq_len, 1)

        _, (h_n, _) = self.lstms(x_node)
        h_last = h_n[-1] # (B*N, hidden_dim)

        out_mean, out_sigma = self.ghead(h_last, -1, self.N) # (B, pred_len, N)
        return out_mean, out_sigma

class TimeFeatureProjection(nn.Module):
    def __init__(self, H, F, C_plus_1, D):
        super().__init__()
        self.feature_proj = nn.Linear(C_plus_1, D)
        self.time_proj = nn.Linear(H, F)

    def forward(self, x):
        x = self.feature_proj(x)
        x = x.permute(0, 2, 1)
        x = self.time_proj(x)
        x = x.permute(0, 2, 1)
        return x
        
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.families = configs.families
        self.families_num = len(self.families)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_nodes = configs.num_nodes
        self.sample_size = configs.sample_size

        dist_info = DISTRIBUTION_REGISTRY[configs.dist_type]
        self.sampler = dist_info["sampler"]
        self.res_dim = [dist_info["res_dim_fn"](len(self.families[i]["children"])) for i in range(self.families_num)]
        
        self.hidden_dim = configs.d_model 
                
        self.lstmforecast = nn.ModuleList([LSTMForecast(len(self.families[i]['children'])+1,
                                                       self.hidden_dim,
                                                       1,
                                                       self.pred_len,
                                                       dropout=0.1) for i in range(self.families_num)])
        
        self.lstmforecastg = LSTMnew(self.hidden_dim,
                                       self.seq_len,
                                       self.pred_len,
                                       self.num_nodes)

        self.multi_fusion = MultiHierarchyFusion(self.num_nodes, 3)

        self.p_block = PeriodBlock(configs)

        self.project_dist = nn.ModuleList([
            TimeFeatureProjection(self.seq_len, self.pred_len, len(self.families[i]["children"]) + 1, self.res_dim[i])
            for i in range(self.families_num)
        ])
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, H, N = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach() # (B, 1, N)
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # (B, 1, N)
        x_enc = x_enc / stdev
        x_enc2 = self.p_block(x_enc, x_mark_enc) # (B, H, d_model)
        family_tensors=[]
        enc_out_list = []
        dist_out_list = []
        for idx, fam in enumerate(self.families):
            p = fam["parent"]
            c = fam["children"]
            index = [p] + c
            index = torch.tensor(
                index, device=x_enc2.device, dtype=torch.long
            )
            # (B, H, N) → (B, H, C+1)
            fam_tensor = x_enc2.index_select(dim=2, index=index)
            family_tensors.append(fam_tensor)
            mean = means.index_select(dim=2, index=index)
            std = stdev.index_select(dim=2, index=index)

            ###################
            # fam_tensor (B, X, C+1)
            enc_out = self.lstmforecast[idx](fam_tensor)
            
            enc_out_list.append(enc_out)
            enc_out = enc_out * std + mean
            self.families[idx]['family_mu'] = enc_out
            ###

            if len(c) > 1:
                dist = self.project_dist[idx](fam_tensor) # (B, F, res_dim)
                dist = F.softplus(dist)
                child_samples = self.sampler(dist) # (B, F, C)
    
                child_samples = child_samples.mean(dim=-2)
                dist_out = enc_out[..., :1] * child_samples # (B, F) * (B, F, C) = (B, F, C)
                dist_out = torch.cat([enc_out[..., 0].unsqueeze(-1), dist_out], dim=-1) # (B, F, C+1)
                self.families[idx]['family_mu_dist'] = dist_out
                self.families[idx]['family_dist'] = dist
            
                dist_out_list.append(dist_out)
            ###################
        enc_out_all, out_sigma = self.lstmforecastg(x_enc2)
        enc_out_empty = torch.zeros(
            (B, self.pred_len, N),
            device=family_tensors[0].device,
            dtype=family_tensors[0].dtype
        )
        enc_out_parent = restore_from_families(self.families, enc_out_list, enc_x=enc_out_empty, mode="parent")
        dist_out_parent = restore_from_families(self.families, dist_out_list, enc_x=enc_out_empty, mode="parent")
        enc_out = self.multi_fusion([enc_out_parent, enc_out_all, dist_out_parent])
        enc_out = enc_out * stdev + means
        out_sigma = out_sigma * stdev
        prob = torch.stack([enc_out, out_sigma], dim=-1) # (B, pred_len, N, 2)
        samples = sampling.generate_normal_samples_nodes(prob) # (B, F, n_samples, N)
        
      
        return self.families, samples, prob




































