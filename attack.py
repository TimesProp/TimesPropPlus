import torch
import torch.nn.functional as F

def pgd_attack(model, x, x_mark, y, epsilon=0.03, step_size=0.005, iters=5, mode="untarget"):
    was_training = model.training
    model.train()
    x_orig = x.clone().detach()
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(iters):
        res = model(x_adv, x_mark, None, None)
        family_output, samples_out = res

        if mode == "inc":
            loss = -samples_out.mean()
        elif mode == "dec":
            loss = samples_out.mean()
        elif mode == "untarget":
            loss = torch.nn.functional.mse_loss(samples_out.mean(dim=2), y)
        else:
            raise ValueError(mode)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad.sign()
            x_adv = x_adv + step_size * grad

            perturb = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = (x_orig + perturb).detach()
            x_adv.requires_grad_(True)
            
    if not was_training:
        model.eval()

    return x_adv.detach()

def pgd_attack1(model, x, x_mark, y, epsilon=0.03, step_size=0.005, iters=20, mode="untarget"):
    was_training = model.training
    model.train()

    x_orig = x.detach().clone()
    x_adv = x.detach().clone().requires_grad_(True)
    x_mark_adv = x_mark.detach().clone()

    for _ in range(iters):
        prob, dist, out, samples_out = model(x_adv, x_mark_adv, None, None)

        if mode == "untarget":
            loss = torch.nn.functional.mse_loss(samples_out.mean(dim=2), y)

        elif mode == "inc":
            loss = -samples_out.mean()                      # increase output

        elif mode == "dec":
            loss = samples_out.mean()                       # decrease output

        elif mode == "inc_dev":
            loss = -(samples_out.mean(dim=2) - y).mean()                # push output > y

        elif mode == "dec_dev":
            loss = (samples_out.mean(dim=2) - y).mean()                 # push output < y

        else:
            raise ValueError(f"Unknown attack mode: {mode}")

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad
        x_adv = x_adv + step_size * grad.sign()

        perturb = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = (x_orig + perturb).detach()
        x_adv.requires_grad_(True)

        x_mark_adv = x_mark.detach().clone()

    if not was_training:
        model.eval()

    return x_adv.detach()


def fgsm_attack(model, x, x_mark, y, epsilon=0.01, mode="untarget"):
    was_training = model.training
    model.train()

    x_adv = x.detach().clone().requires_grad_(True)
    x_mark_adv = x_mark.detach().clone()

    prob, dist, out, samples_out = model(x_adv, x_mark_adv, None, None)

    if mode == "untarget":
        loss = F.mse_loss(out, y)

    elif mode == "inc":
        loss = -out.mean()

    elif mode == "dec":
        loss = out.mean()

    elif mode == "inc_dev":
        loss = F.mse_loss(out, y) + 0.01 * out.mean()

    elif mode == "dec_dev":
        loss = F.mse_loss(out, y) - 0.01 * out.mean()

    else:
        raise ValueError(f"Unknown FGSM mode: {mode}")

    model.zero_grad()
    loss.backward()

    grad = x_adv.grad
    x_adv = x_adv + epsilon * grad.sign()

    x_adv = x_adv.detach()

    if not was_training:
        model.eval()

    return x_adv

def scaling_attack(x, p_te=0.5, epsilon_sa=0.2, mode="both"):
    if mode == "pos":
        lambda_min = 1.0
        lambda_max = 1.0 + epsilon_sa
    elif mode == "neg":
        lambda_min = 1.0 - epsilon_sa
        lambda_max = 1.0
    elif mode == "both":
        lambda_min = 1.0 - epsilon_sa
        lambda_max = 1.0 + epsilon_sa
    else:
        raise ValueError("Unknown mode for scaling attack")
    x_adv = x.detach().clone()
    mask = (torch.rand(1, 1, x.shape[-1], device=x.device) < p_te)
    lambdas = torch.empty(1, 1, x.shape[-1], device=x.device).uniform_(lambda_min, lambda_max)
    x_adv = torch.where(mask, x_adv * lambdas, x_adv)

    return x_adv
