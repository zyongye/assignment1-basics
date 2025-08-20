import math

import torch
import torch.nn as nn

from einops import einsum

class Linear(nn.Module):

    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(nn.init.trunc_normal_(
                                    torch.empty(d_out, d_in, device=device, dtype=dtype),
                                    mean = 0,
                                    std=std,
                                    a=-3*std,
                                    b=3*std,),
        requires_grad=True,)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3, b=3,
            ),
            requires_grad=True,
        )

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.scale = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
            requires_grad=True
        )
        self.inv_d_model = 1 / d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_in=d_model, d_out=d_ff)
        self.w2 = Linear(d_in=d_ff, d_out=d_model)
        self.w3 = Linear(d_in=d_model, d_out=d_ff)
    
    def forward(self, x: torch.Tensor):
        x_gated = self.w1(x)
        x_linear = self.w3(x)
        x_swiglu = silu(x_gated) * x_linear
        return self.w2(x_swiglu)





