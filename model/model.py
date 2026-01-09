import math
import torch.nn as nn
import torch
from transformers import PretrainedConfig
from typing import Optional


class MokioMindConfig(PretrainedConfig):
    model_type = " minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,

        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# 继承nn.Module

class RMSNorm(nn.Module):
    # _init_初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

# _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# forward
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)*x


# 写出ROPE的公式
def precomput_feqs_cis(dim: int, end: int = int(32*1024), rope_base: float = 1e6,
                       rope_scaling: Optional[dict] = None):
    # 写出ROPE的公式
    freqs = 1.0/(rope_base**torch.arange(0, dim, 2)[:dim//2].float()/dim)
    # 计算inv_freq

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        # 计算corr_dim
        corr_dim = next((i for i in range(dim//2) if 2 *
                        math.pi/freqs[i] > orig_max), dim//2)
        # 计算power
        power = torch.arange(
            0, dim//2, device=freqs.device).float()/(max(1, dim//2-1))
        # 计算beta
        beta = beta_slow + (beta_fast - beta_slow)*power
        # 计算scale
        scale = torch.where(
            torch.arange(0, dim//2, device=freqs.device) < corr_dim,
            (beta*factor-beta+1)/beta*factor,
            1.0/factor
        )
        # 应用scale
        freqs = freqs*scale
   # 生成位置索引
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
 # 返回一个cos和sin
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # [a,b] => [-b,a]
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    # unsqueeze cos and sin
    q_embed = (q*cos.unsqueeze(unsqueeze_dim)) + \
        (rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim)) + \
        (rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int):
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
# x[:, :, :, None, :]= x.unsqueeze(3)
# shape=(bs, slen, num_key_value_heads, 1, head_dim)
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads*n_rep, head_dim)
            )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (args.num_key_value_heads if args.num_key_value_heads is
                                    not None else args.num_attention_heads)
        assert args.hidden_size % args.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            args.num_attention_heads*self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention
