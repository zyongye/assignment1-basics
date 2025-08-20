import math

import torch
import torch.nn as nn

from einops import einsum

class Linear(nn.Module):

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(nn.init.trunc_normal_(
                                    torch.empty(d_out, d_in),
                                    mean = 0,
                                    std=std,
                                    a=-3*std,
                                    b=3*std,),
        requires_grad=True,)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")








