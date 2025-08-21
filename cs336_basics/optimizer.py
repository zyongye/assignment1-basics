import math
from typing import Optional, Callable

import torch


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
            "lr": lr,
            "beta": betas, 
            "eps": eps,
            "gamma": weight_decay,   
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            gamma = group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.grad.data))
                v = state.get("v", torch.zeros_like(p.grad.data))
                m_t = beta[0] * m + (1 - beta[0]) * p.grad.data
                v_t = beta[1] * v + (1 - beta[1]) * torch.square(p.grad.data)
                lr_t = lr * math.sqrt(1 - beta[1] ** t) / (1 - beta[0] ** t)
                p.data -= lr_t * m_t / (torch.sqrt(v_t) + eps)
                p.data -= lr * gamma * p.data
                state["t"] = t + 1
                state["m"] = m_t
                state["v"] = v_t
        return loss


