import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call
import torch.nn.functional as F
import copy

class B_SimpleNN(nn.Module):
    def __init__(self, base_model, prior_std=2.0):
        super().__init__()
        self.base_model = base_model
        self.prior_std = prior_std
        self.name_map = {}

        self.mu = nn.ParameterDict()
        self.rho = nn.ParameterDict()

        for name, param in base_model.named_parameters():
            safe_name = name.replace(".", "_")  
            self.name_map[safe_name] = name

            self.mu[safe_name] = nn.Parameter(param.data.clone())
            self.rho[safe_name] = nn.Parameter(torch.full_like(param.data, -5.0))

    def sample_weights(self):
        sampled = {}
        for safe_name, real_name in self.name_map.items():
            sigma = F.softplus(self.rho[safe_name])
            eps = torch.randn_like(sigma)
            sampled[real_name] = self.mu[safe_name] + sigma * eps
        return sampled

    def kl_divergence(self):
        all_mu = []
        all_rho = []
        
        for safe_name in self.name_map.keys():
            all_mu.append(self.mu[safe_name].view(-1))
            all_rho.append(self.rho[safe_name].view(-1))
            
        mu = torch.cat(all_mu)
        rho = torch.cat(all_rho)
        sigma = F.softplus(rho)
        
        prior_sigma = self.prior_std

        kl = (torch.log(prior_sigma / (sigma + 1e-8))
              + (sigma**2 + mu**2) / (2 * prior_sigma**2)
              - 0.5).sum()
        return kl

    def forward(self, *args, **kwargs):
        sampled_params = self.sample_weights()
        families, samples, prob = functional_call(self.base_model, sampled_params, args, kwargs)
        return families, samples, prob

    def forward_multi(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, n_samples=10):
        families_out, samples_sum, prob_sum = None, None, None
        
        for _ in range(n_samples):
            families, samples, prob = self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if samples_sum is None:
                families_out = families
                samples_sum = samples
                prob_sum = prob
            else:
                samples_sum += samples
                prob_sum += prob

        return families_out, samples_sum / n_samples, prob_sum / n_samples

    def bayes_forward(self, *args, K=10, **kwargs):
        families_out, samples_sum, prob_sum = None, None, None

        for _ in range(K):
            families, samples, prob = self.forward(*args, **kwargs)

            if samples_sum is None:
                families_out = families
                samples_sum = samples
                prob_sum = prob
            else:
                samples_sum += samples
                prob_sum += prob

        return families_out, samples_sum / K, prob_sum / K