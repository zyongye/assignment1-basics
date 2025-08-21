import math

import torch
import torch.nn as nn
import einx

from einops import einsum, rearrange

from .util import softmax


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

"""
Q: why we need normalization in general:
A: prevent gradient from vanish or explode, stablize training
   smoothier optimization landscape
Q: Pre-layer norm vs post-layer norm:
A: More stable gradient
"""
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
    
class RoPE(nn.Module):

    def __init__ (self, theta:float, d_k:int, max_seq_len: int, device=None):
        super().__init__()
        self.register_buffer(
            "_rope_buffer", RoPE._init_buffer(
                theta, d_k, max_seq_len
            ),
            persistent=False
        )

    @staticmethod
    def _init_buffer(theta, d_k, max_seq_len):
        d = torch.arange(0, d_k, 2) / d_k
        freqs = theta ** -d
        t = torch.arange(max_seq_len)

        freqs = einsum(t, freqs, "t, f -> t f")
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        og_shape = x.shape
        cos, sin = self._rope_buffer[:, token_positions, :]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        return torch.stack((x1_rot, x2_rot), dim=-1).view(og_shape)

def sdpa(q, k, v, mask=None):
    scale = math.sqrt(q.shape[-1])
    a_score = einsum(q, k, "batch ... seq_q dk, batch ... seq_k dk -> batch ... seq_q seq_k")
    mask_a = torch.where(mask, a_score, torch.tensor(float('-inf')))
    a_score = softmax(mask_a / scale)
    return einsum(a_score, v, "batch ... seq_q seq_k, batch ... seq_k dv -> batch ... seq_q dv")



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, has_rope=False, max_seq_len=0, rope_theta=0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv = Linear(d_model, 3 * d_model)
        self.o_proj = Linear(d_model, d_model)
        if has_rope:
            self.rope = RoPE(rope_theta, d_model // num_heads, max_seq_len)


    def forward(self, x, token_positions=None, apply_rope=False):
        B,S,_ = x.shape
        qkv = self.qkv(x)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
        q = qkv[..., :self.d_model].view(B, S, self.num_heads, -1).transpose(1, 2)
        if apply_rope:
            q = self.rope(q, token_positions)
        k = qkv[..., self.d_model:2 * self.d_model].view(B, S, self.num_heads, -1).transpose(1, 2)
        if apply_rope:
            k = self.rope(k, token_positions)
        v = qkv[..., 2 * self.d_model: 3 * self.d_model].view(B, S, self.num_heads, -1).transpose(1, 2)

        o = sdpa(q, k, v, mask).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(o)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.pre_attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, has_rope=True, max_seq_len=max_seq_len, rope_theta=theta)
        self.post_attn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def weight_loader(self, weights):
        self.pre_attn_norm.scale.data = weights["ln1.weight"]
        self.post_attn_norm.scale.data = weights["ln2.weight"]
        self.ffn.w1.weight.data = weights["ffn.w1.weight"]
        self.ffn.w2.weight.data = weights["ffn.w2.weight"]
        self.ffn.w3.weight.data = weights["ffn.w3.weight"]
        self.attn.o_proj.weight.data = weights["attn.output_proj.weight"]
        q = weights["attn.q_proj.weight"]
        k = weights["attn.k_proj.weight"]
        v = weights["attn.v_proj.weight"]
        qkv = torch.cat([q, k, v], dim=0)
        self.attn.qkv.weight.data = qkv
    
    def forward(self, x: torch.Tensor, positions=None):
        B, S, _ = x.shape
        if not positions:
            positions = torch.arange(S).unsqueeze(0).expand(B, -1)
        x_attn = self.attn(self.pre_attn_norm(x), token_positions=positions, apply_rope=True)
        x = x + x_attn
        x_ffn = self.ffn(self.post_attn_norm(x))
        x = x + x_ffn
        return x
    
class TransformerForCausalLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.num_layers = num_layers
        self.token_embedding = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for 
                      i in range(num_layers)]
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def weight_loader(self, weights):
        self.token_embedding.weight.data = weights["token_embeddings.weight"]
        self.final_norm.scale.data = weights["ln_final.weight"]
        self.lm_head.weight.data = weights["lm_head.weight"]
        for i in range(self.num_layers):
            per_layer_weights = {
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            }
            self.blocks[i].weight_loader(per_layer_weights)
    
    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)




