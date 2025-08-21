import math
import torch


def softmax(x: torch.Tensor, dim=-1):
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    x = x / x.sum(dim=dim, keepdim=True)
    return x

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    x = -log_softmax(inputs)
    return torch.mean(torch.gather(x, -1, targets.unsqueeze(-1)))


def cos_anneling_lr_decay(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t / T_w * alpha_max
    if t > T_c:
        return alpha_min
    return alpha_min + (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min) / 2

def gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    eps = 10e-6
    norm = 0
    for g in grads:
        norm += (g**2).sum()
    # require norm for all params
    norm = torch.sqrt(norm)
    if norm > max_l2_norm:
        factor = max_l2_norm / (norm + eps)
        for g in grads:
            g *= factor






