import torch
import torch.nn as nn
from scipy.special import gammaln

def dirichlet_loss(data, params, eps=1e-6, debug=False):
    def beta_function_log(beta):
        log_gamma_sum = torch.lgamma(beta).sum(dim=-1)
        log_gamma_total = torch.lgamma(beta.sum(dim=-1))
        return log_gamma_sum - log_gamma_total

    C_true = data
    C_pred = params # (B, F, C)
    first_term = (C_pred - 1.0) * torch.log(torch.clamp(C_true, min=eps))
    first_term_sum = -first_term.sum(dim=-1)
    second_term = beta_function_log(C_pred)

    loss = (first_term_sum + second_term).mean()

    if debug:
        print("C_true[0, 0]:", C_true[0, 0])
        print("C_pred[0, 0]:", C_pred[0, 0])
        print("first_term_sum[0, 0]:", first_term_sum[0, 0])
        print("second_term[0, 0]:", second_term[0, 0])
        print("loss:", loss.item())

    return loss
    
def gdd_loss(true_values, params, eps=1e-6, debug=False):
    def compute_log_likelihood(true_values, params):
        B, F, C = true_values.shape
        C_minus_1 = C - 1
        params = params.view(B, F, C_minus_1, 2)

        alpha = params[..., 0]  # (B, F, C-1)
        beta = params[..., 1]   # (B, F, C-1)

        alpha = torch.clamp(alpha, min=eps, max=1e4)
        beta = torch.clamp(beta, min=eps, max=1e4)

        def smooth_target(data, epsilon=1e-2):
            C = data.shape[-1]
            return (1 - epsilon) * data + epsilon / C

        true_values = smooth_target(true_values, epsilon=0.2)
        true_values = torch.clamp(true_values, min=eps, max=1.0)

        p = true_values[..., :-1]   # (B, F, C-1)
        p_k = true_values[..., -1:]  # (B, F, 1)

        log_p = torch.log(p + eps)         # (B, F, C-1)
        log_p_k = torch.log(p_k + eps)     # (B, F, 1)

        cumulative_sum = torch.cumsum(p.flip(-1), dim=-1).flip(-1) + p_k
        cumulative_sum = torch.clamp(cumulative_sum, min=eps, max=1e6)
        log_cumulative_sum = torch.log(cumulative_sum)

        beta_log_terms = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        beta_log_sum = beta_log_terms.sum(dim=-1)

        alpha_log_p = ((alpha - 1) * log_p).sum(dim=-1)

        b_i_minus_1 = torch.cat([torch.zeros_like(beta[..., :1]), beta[..., :-1]], dim=-1)
        beta_log_sum_p = (b_i_minus_1 - (alpha + beta)) * log_cumulative_sum
        beta_log_sum_p = beta_log_sum_p.sum(dim=-1)

        p_k_term = (beta[..., -1:] - 1) * log_p_k
        p_k_term = p_k_term.squeeze(-1)

        log_likelihood = -beta_log_sum + alpha_log_p + beta_log_sum_p + p_k_term
        return log_likelihood, {
            "alpha": alpha,
            "beta": beta,
            "p": p,
            "p_k": p_k,
            "log_p": log_p,
            "log_cumulative_sum": log_cumulative_sum,
            "beta_log_sum": beta_log_sum,
            "alpha_log_p": alpha_log_p,
            "beta_log_sum_p": beta_log_sum_p,
            "p_k_term": p_k_term,
            "log_likelihood": log_likelihood
        }

    log_likelihood, ctx = compute_log_likelihood(true_values, params)

    valid_mask = (~torch.isnan(log_likelihood)) & (~torch.isinf(log_likelihood))
    if valid_mask.sum() == 0:
        return torch.tensor(1e6, device=true_values.device)

    loss = -log_likelihood[valid_mask].mean()
    return loss


def beta_liouville_loss(true_values, params, eps=1e-6, debug=False):
    B, F, C = true_values.shape

    alpha_params = params[:, :, :C - 1]         # shape (B, F, C-1)
    alpha = params[:, :, C - 1:C]               # shape (B, F, 1)
    beta = params[:, :, C:C + 1]                # shape (B, F, 1)

    true_sum = true_values[:, :, :-1].sum(dim=-1, keepdim=True)
    true_sum = torch.clamp(true_sum, min=eps)

    log_gamma_sum_alpha = torch.lgamma(alpha_params.sum(dim=-1, keepdim=True))
    log_gamma_alpha_beta = torch.lgamma(alpha + beta)
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_beta = torch.lgamma(beta)
    log_gamma_individual = torch.lgamma(alpha_params).sum(dim=-1, keepdim=True)

    log_true_values = ((alpha_params - 1) * torch.log(torch.clamp(true_values[:, :, :-1], min=eps))).sum(dim=-1, keepdim=True)

    log_sum_term = (alpha - alpha_params.sum(dim=-1, keepdim=True)) * torch.log(true_sum)

    log_one_minus_sum_term = (beta - 1) * torch.log(torch.clamp(true_values[:, :, -1].unsqueeze(-1), min=eps))

    log_likelihood = (
        log_gamma_sum_alpha + log_gamma_alpha_beta
        - log_gamma_alpha - log_gamma_beta - log_gamma_individual
        + log_true_values + log_sum_term + log_one_minus_sum_term
    )

    loss = -log_likelihood.mean()

    if debug:
        print("alpha_params[0, 0]:", alpha_params[0, 0])
        print("alpha[0, 0]:", alpha[0, 0])
        print("beta[0, 0]:", beta[0, 0])
        print("log_likelihood[0, 0]:", log_likelihood[0, 0])
        print("loss:", loss.item())

    return loss


def scaled_dirichlet_loss(data, params, eps=1e-6, debug=False):
    B, F, D = data.shape
    alpha = params[:, :, :D]
    beta = params[:, :, D:]

    alpha = torch.clamp(alpha, min=eps, max=1e4)
    beta = torch.clamp(beta, min=eps, max=1e4)

    def smooth_target(x, epsilon=1e-2):
        C = x.shape[-1]
        return (1 - epsilon) * x + epsilon / C

    data = smooth_target(data, epsilon=0.01)
    data = torch.clamp(data, min=eps, max=1.0)

    alpha_plus = torch.sum(alpha, dim=-1, keepdim=True)
    alpha_plus = torch.clamp(alpha_plus, min=1e-3, max=1e4)

    log_gamma_alpha_plus = torch.lgamma(alpha_plus)
    log_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)

    log_beta_term = torch.sum(alpha * torch.log(beta + eps), dim=-1, keepdim=True)
    log_x_term = torch.sum((alpha - 1) * torch.log(data + eps), dim=-1, keepdim=True)

    sum_beta_x = torch.sum(beta * data, dim=-1, keepdim=True)
    sum_beta_x = torch.clamp(sum_beta_x, min=eps, max=1e6)
    log_norm_term = alpha_plus * torch.log(sum_beta_x)

    log_likelihood = (
        log_gamma_alpha_plus - log_gamma_alpha
        + log_beta_term + log_x_term - log_norm_term
    )

    valid_mask = (~torch.isnan(log_likelihood)) & (~torch.isinf(log_likelihood))
    if valid_mask.sum() == 0:
        return torch.tensor(1e6, device=data.device)
    return -log_likelihood[valid_mask].mean()

def shifted_scaled_dirichlet_loss(data, params, eps=1e-6, debug=True):
    def smooth_target(data, epsilon=1e-2):
        C = data.shape[-1]
        return (1 - epsilon) * data + epsilon / C
    
    b, F, param_size = params.shape
    C = (param_size - 1) // 2

    alpha = params[..., :C]
    scale_factors = params[..., C:2 * C]
    a = params[..., -1].unsqueeze(-1)

    alpha = torch.clamp(alpha, min=1e-3, max=1e3)
    scale_factors = torch.clamp(scale_factors, min=1e-3, max=1e3)
    a = torch.clamp(a, min=0.2, max=1e2)

    data = smooth_target(data, epsilon=0.6) # 0.2
    data = torch.clamp(data, min=eps, max=1.0)

    ratio = torch.clamp(alpha / a, min=eps, max=1e4) # max=1e4
    alpha_plus = torch.sum(alpha, dim=-1, keepdim=True)

    log_gamma_alpha_plus = torch.lgamma(alpha_plus)
    log_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)

    log_a_term = (C - 1) * torch.log(a + eps)
    log_p_term = torch.sum(ratio * torch.log(scale_factors + eps), dim=-1, keepdim=True)
    log_x_term = torch.sum((ratio - 1) * torch.log(data + eps), dim=-1, keepdim=True)

    power = (1.0 / (a + eps)) * torch.log((data + eps) / (scale_factors + eps))
    log_base = torch.log((data + eps) / (scale_factors + eps))
    log_trans = (1.0 / (a + eps)) * log_base
    log_trans_max, _ = torch.max(log_trans, dim=-1, keepdim=True)
    log_sum_trans = log_trans_max + torch.log(torch.sum(torch.exp(log_trans - log_trans_max), dim=-1, keepdim=True) + eps)
    log_sum_term = alpha_plus * log_sum_trans

    log_likelihoods = (
        log_gamma_alpha_plus - log_gamma_alpha - log_a_term - log_p_term + log_x_term - log_sum_term
    )

    valid_mask = (~torch.isnan(log_likelihoods)) & (~torch.isinf(log_likelihoods))
    if valid_mask.sum() == 0:
        return torch.tensor(1e6, device=data.device)
    return -log_likelihoods[valid_mask].mean()
