"""
ViT Decoder with Mixture-of-Experts Adapters.

Architecture (mirrored from MoECL / CLIP VisualTransformer):
  - Patch projection: linear, maps spatial feature channels -> hidden_dim
  - Positional embedding (learnable)
  - N x ResidualAttentionBlock (shared self-attn + layer-norm + MoE FFN adapters)
  - Reconstruction head: linear, maps hidden_dim -> original channels
  - (Optional) Conv up-sample to original spatial resolution

Input dict keys consumed:  "feature_align"  (B, C, H, W)
Output dict keys produced: "feature_rec"     (B, C, H, W)
                           "image_rec"        (B, C, H, W)  -- alias for compatibility
                           "moe_loss"         scalar tensor (load-balance loss)
"""

from collections import OrderedDict, Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .adapter import Adapter
from models.initializer import initialize_from_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Minimal argument container (replaces argparse.Namespace used in MoECL)
# ─────────────────────────────────────────────────────────────────────────────

class _Args:
    """Simple namespace so ResidualAttentionBlock can be reused without argparse."""
    def __init__(self, task_id, experts_num, topk, ffn_num,
                 apply_moe, ffn_adapt, ffn_option, autorouter, is_train):
        self.task_id    = task_id
        self.experts_num = experts_num
        self.topk       = topk
        self.ffn_num    = ffn_num           # adapter bottleneck dim
        self.apply_moe  = apply_moe
        self.ffn_adapt  = ffn_adapt
        self.ffn_option = ffn_option        # 'parallel'
        self.autorouter = autorouter
        self.is_train   = is_train


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Dispatcher  (unchanged from MoECL reference)
# ─────────────────────────────────────────────────────────────────────────────

class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates       = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes  = (gates > 0).sum(0).tolist()
        gates_exp         = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros    = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                               device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level building blocks
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        return super().forward(x.float()).type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Attention Block with MoE Adapters
# ─────────────────────────────────────────────────────────────────────────────

class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with optional parallel MoE adapter FFN.
    Mirrors the MoECL ResidualAttentionBlock exactly.
    """

    def __init__(self, d_model: int, n_head: int,
                 attn_mask: Optional[torch.Tensor] = None,
                 adapter_flag: bool = False,
                 args: Optional[_Args] = None,
                 layer_idx: int = 0):
        super().__init__()

        self.attn     = nn.MultiheadAttention(d_model, n_head)
        self.ln_1     = LayerNorm(d_model)
        self.mlp      = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, d_model * 4)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2     = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.layer         = layer_idx
        self.adapter_flag  = adapter_flag
        self.adaptmlp_list = nn.ModuleList()

        if args is None:
            return  # bare transformer block, no adapters

        self.task_id      = args.task_id
        self.experts_num  = args.experts_num
        self.top_k        = args.topk
        self.ffn_adapt    = args.ffn_adapt
        self.ffn_option   = args.ffn_option   # 'parallel'
        self.ffn_num      = args.ffn_num
        self.apply_moe    = args.apply_moe
        self.autorouter   = args.autorouter
        self.is_train     = args.is_train
        self.noisy_gating = True
        self.noise_epsilon = 1e-2
        self.d_model      = d_model

        self.softmax  = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std",  torch.tensor([1.0]))

        # Usage counters (no grad)
        self.choose_map = torch.zeros([self.experts_num])

        if self.ffn_adapt and self.adapter_flag:
            if self.apply_moe:
                if self.task_id > -1:
                    # Per-task routers
                    self.router_list  = nn.ParameterList()
                    self.w_noise_list = nn.ParameterList()
                    for _ in range(11):   # max task slots
                        self.router_list.append(
                            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                        self.w_noise_list.append(
                            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                else:
                    # Single shared router
                    self.router1  = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
                    self.w_noise  = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)

                for _ in range(self.experts_num):
                    self.adaptmlp_list.append(
                        Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                init_option='lora', adapter_scalar=0.1,
                                adapter_layernorm_option='none'))
            else:
                # Single adapter, no MoE
                self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                        init_option='lora', adapter_scalar=0.1,
                                        adapter_layernorm_option='none')

    # ── gating helpers ────────────────────────────────────────────────────────

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m     = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in  = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in  = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in            = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in  = normal.cdf((clean_values - threshold_if_in)  / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev     = self.softplus(raw_noise_stddev) + noise_epsilon
            logits           = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
            noise_stddev = None

        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits  = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates   = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = self._prob_in_top_k(
                clean_logits, logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    # ── attention helper ──────────────────────────────────────────────────────

    def attention(self, x: torch.Tensor):
        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    # ── forward ───────────────────────────────────────────────────────────────

    def _run_moe(self, x, w_gate, w_noise):
        """Dispatch tokens through MoE experts and return combined output + loss."""
        # Use [CLS] token (position 0) as routing signal
        x_re = x.permute(1, 0, 2)[:, 0, :]   # (B, D)

        gates, load = self.noisy_top_k_gating(x_re, self.is_train, w_gate, w_noise)
        importance  = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * 1e-2

        # Track expert usage
        nonzero_indices = torch.nonzero(gates)
        counter = Counter(nonzero_indices[:, 1].tolist())
        for number, count in counter.items():
            self.choose_map[number] += count

        dispatcher    = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(
            x.permute(1, 0, 2).view(x.shape[1], -1))   # (B, L*D)

        expert_outputs = [
            self.adaptmlp_list[i](
                expert_inputs[i].view(expert_inputs[i].shape[0],
                                      x.shape[0], x.shape[2]).to(x),
                add_residual=False)
            for i in range(self.experts_num)
        ]

        # Drop empty expert slots
        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0:
                expert_outputs.pop(i)
            else:
                expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                i += 1

        y = dispatcher.combine(expert_outputs)               # (B, L*D)
        y = y.view(x.shape[1], x.shape[0], x.shape[2])      # (B, L, D)
        return y.permute(1, 0, 2), loss                      # (L, B, D), scalar

    def forward(self, x: torch.Tensor):
        """
        x: (L, B, D)  — sequence-first layout as used by nn.MultiheadAttention
        Returns: (L, B, D), moe_loss scalar
        """
        x = x + self.attention(self.ln_1(x))

        moe_loss = torch.tensor(0.0, device=x.device)

        has_moe = (hasattr(self, 'ffn_adapt') and self.ffn_adapt
                   and self.ffn_option == 'parallel'
                   and self.adapter_flag)

        if has_moe:
            if self.apply_moe:
                if self.task_id > -1:
                    w_gate  = self.router_list[self.task_id]
                    w_noise = self.w_noise_list[self.task_id]
                else:
                    w_gate  = self.router1
                    w_noise = self.w_noise

                adapter_out, moe_loss = self._run_moe(x, w_gate, w_noise)
                x = x + self.mlp(self.ln_2(x)) + adapter_out
            else:
                x_re    = x.permute(1, 0, 2)
                adapt_x = self.adaptmlp(x_re, add_residual=False).permute(1, 0, 2)
                x = x + self.mlp(self.ln_2(x)) + adapt_x
        else:
            x = x + self.mlp(self.ln_2(x))

        return x, moe_loss


# ─────────────────────────────────────────────────────────────────────────────
# Transformer stack (wraps N ResidualAttentionBlocks)
# ─────────────────────────────────────────────────────────────────────────────

class ViTDecoderTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: Optional[torch.Tensor] = None,
                 adapter_flag: bool = True,
                 args: Optional[_Args] = None):
        super().__init__()
        self.width  = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask,
                                   adapter_flag, args, i)
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        total_loss = torch.tensor(0.0, device=x.device)
        for blk in self.resblocks:
            x, loss = blk(x)
            total_loss = total_loss + loss
        return x, total_loss

    def get_experts(self):
        """Return per-expert lists of adapters across all layers (for LocalityLoss)."""
        if not self.resblocks[0].adaptmlp_list:
            return []
        num_experts = self.resblocks[0].experts_num
        expert_groups = []
        for m in range(num_experts):
            expert_groups.append(
                nn.ModuleList([blk.adaptmlp_list[m] for blk in self.resblocks
                               if len(blk.adaptmlp_list) > m]))
        return expert_groups


# ─────────────────────────────────────────────────────────────────────────────
# ViT Decoder (drop-in replacement for old ResNet-based VisDecoder)
# ─────────────────────────────────────────────────────────────────────────────

class ViTDecoder(nn.Module):
    """
    ViT-based feature decoder with MoE adapters.

    Parameters
    ----------
    inplanes   : list[int]  — [C]  input feature channels from backbone
    instrides  : list[int]  — [S]  stride of the backbone feature map
    hidden_dim : int        — transformer (and adapter) width
    num_layers : int        — number of transformer blocks
    num_heads  : int        — attention heads (hidden_dim must be divisible)
    num_experts: int        — number of MoE adapter experts per block
    topk       : int        — top-k routing per token
    ffn_num    : int        — adapter bottleneck dimension
    task_id    : int        — current task index (-1 = shared router)
    apply_moe  : bool       — enable MoE routing
    autorouter : bool       — auto-select router at inference
    initializer: cfg        — weight initializer config (optional)
    """

    def __init__(self,
                 inplanes,
                 instrides,
                 hidden_dim: int = 256,
                 num_layers:  int = 4,
                 num_heads:   int = 8,
                 num_experts: int = 12,
                 topk:        int = 2,
                 ffn_num:     int = 64,
                 task_id:     int = -1,
                 apply_moe:   bool = True,
                 autorouter:  bool = False,
                 initializer=None,
                 **kwargs):
        super().__init__()

        assert isinstance(inplanes,  list) and len(inplanes)  == 1
        assert isinstance(instrides, list) and len(instrides) == 1

        in_ch   = inplanes[0]
        stride  = instrides[0]

        # ── input projection: channels -> hidden_dim ──────────────────────────
        self.input_proj = nn.Linear(in_ch, hidden_dim)

        # ── CLS token + positional embedding (computed lazily) ────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── args object for ResidualAttentionBlock ─────────────────────────────
        args = _Args(
            task_id    = task_id,
            experts_num = num_experts,
            topk       = topk,
            ffn_num    = ffn_num,
            apply_moe  = apply_moe,
            ffn_adapt  = True,
            ffn_option = 'parallel',
            autorouter = autorouter,
            is_train   = True,       # toggled in train()/eval() via self.training property below
        )
        self._args = args

        # ── transformer blocks ────────────────────────────────────────────────
        self.transformer = ViTDecoderTransformer(
            width       = hidden_dim,
            layers      = num_layers,
            heads       = num_heads,
            adapter_flag = True,
            args        = args,
        )

        # ── output projection: hidden_dim -> original channels ────────────────
        self.output_proj = nn.Linear(hidden_dim, in_ch)
        self.ln_post     = LayerNorm(hidden_dim)

        # ── up-sample to original feature stride ──────────────────────────────
        self.upsample = (nn.UpsamplingBilinear2d(scale_factor=stride)
                         if stride > 1 else nn.Identity())

        # ── positional embedding cache (filled on first forward) ───────────────
        self._pos_embed: Optional[torch.Tensor] = None
        self._seq_len   = -1

        initialize_from_cfg(self, initializer)

    # ── sync is_train flag with PyTorch train/eval mode ──────────────────────

    def train(self, mode: bool = True):
        self._args.is_train = mode
        for blk in self.transformer.resblocks:
            if hasattr(blk, 'is_train'):
                blk.is_train = mode
        return super().train(mode)

    # ── task-id setter (called by training loop) ──────────────────────────────

    def set_task_id(self, task_id: int):
        self._args.task_id = task_id
        for blk in self.transformer.resblocks:
            if hasattr(blk, 'task_id'):
                blk.task_id = task_id

    # ── expert access for LocalityLoss ─────────────────────────────────────────

    def get_experts(self):
        return self.transformer.get_experts()

    # ── positional embedding ──────────────────────────────────────────────────

    def _get_pos_embed(self, seq_len: int, hidden_dim: int, device):
        """Sinusoidal positional embedding, cached."""
        if self._pos_embed is not None and self._seq_len == seq_len:
            return self._pos_embed.to(device)
        pos  = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        dim  = torch.arange(hidden_dim, dtype=torch.float32, device=device).unsqueeze(0)
        ang  = pos / (10000 ** (2 * (dim // 2) / hidden_dim))
        pe   = torch.zeros(seq_len, hidden_dim, device=device)
        pe[:, 0::2] = ang[:, 0::2].sin()
        pe[:, 1::2] = ang[:, 1::2].cos()
        self._pos_embed = pe
        self._seq_len   = seq_len
        return pe   # (L, D)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, input: dict) -> dict:
        """
        input["feature_align"]: (B, C, H, W)
        Returns dict with keys: "feature_rec", "image_rec", "moe_loss"
        """
        feat = input["feature_align"]   # (B, C, H, W)
        B, C, H, W = feat.shape
        L = H * W                       # sequence length (patch tokens)

        # Flatten spatial dims -> token sequence: (B, L, C)
        tokens = feat.flatten(2).permute(0, 2, 1)   # (B, L, C)

        # Project to hidden_dim
        tokens = self.input_proj(tokens)             # (B, L, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, L+1, D)

        # Add sinusoidal positional embedding
        pe     = self._get_pos_embed(L + 1, tokens.shape[-1], tokens.device)  # (L+1, D)
        tokens = tokens + pe.unsqueeze(0)            # (B, L+1, D)

        # Convert to sequence-first for nn.MultiheadAttention
        tokens = tokens.permute(1, 0, 2)             # (L+1, B, D)

        # Transformer forward
        tokens, moe_loss = self.transformer(tokens)  # (L+1, B, D), scalar

        # Drop CLS token, back to (B, L, D)
        tokens = tokens[1:].permute(1, 0, 2)         # (B, L, D)
        tokens = self.ln_post(tokens)

        # Project back to original channel dim
        feature_rec = self.output_proj(tokens)        # (B, L, C)

        # Reshape to spatial: (B, C, H, W)
        feature_rec = feature_rec.permute(0, 2, 1).view(B, C, H, W)

        return {
            "feature_rec": feature_rec,
            "image_rec":   feature_rec,   # alias kept for downstream compatibility
            "moe_loss":    moe_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory (mirrors old VisDecoder() signature)
# ─────────────────────────────────────────────────────────────────────────────

def VisDecoder(**kwargs):
    """
    Factory function.  All keyword arguments are forwarded to ViTDecoder.
    The old 'block_type' argument is silently ignored for backward compat.
    """
    kwargs.pop("block_type", None)
    kwargs.pop("layers",     None)
    return ViTDecoder(**kwargs)
