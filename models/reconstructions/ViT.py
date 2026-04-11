import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % self.num_heads == 0, "embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, h, w, _ = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = rearrange(q, 'b h w (n d) -> b n h w d', n=self.num_heads)
        k = rearrange(k, 'b h w (n d) -> b n h w d', n=self.num_heads)
        v = rearrange(v, 'b h w (n d) -> b n h w d', n=self.num_heads)

        k_T = k.transpose(-2, -1)
        scores = torch.matmul(q, k_T) / self.head_dim**0.5
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)

        out = rearrange(out, 'b n h w d -> b h w (n d)')
        out = self.out(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = rearrange(x, 'b h w d -> b (h w) d')
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(x.size(1) ** 0.5))
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, num_heads, mlp_dim):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        out = self.self_attention(self.norm1(x))
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Mixture of Experts Components
# ============================================================

class MoERouter(nn.Module):
    """
    Top-K gating network for routing inputs to experts.
    Takes global-pooled stem features and produces per-expert probabilities.
    """
    def __init__(self, input_dim, num_experts, top_k=1):
        super(MoERouter, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Args:
            x: (B, D) global-pooled stem features
        Returns:
            top_k_probs: (B, top_k) normalized weights for selected experts
            top_k_indices: (B, top_k) indices of selected experts
            all_probs: (B, num_experts) full softmax probabilities (for loss)
        """
        logits = self.gate(x)  # (B, num_experts)
        all_probs = F.softmax(logits, dim=-1)  # (B, num_experts)
        top_k_probs, top_k_indices = torch.topk(all_probs, self.top_k, dim=-1)
        # Renormalize top-k probs to sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return top_k_probs, top_k_indices, all_probs


class Expert(nn.Module):
    """
    A single expert consisting of multiple ViTBlocks.
    Each expert specializes in reconstructing features for specific class(es).
    """
    def __init__(self, hidden_dim, num_patches, num_heads, mlp_dim, num_blocks=4):
        super(Expert, self).__init__()
        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, num_patches, hidden_dim, num_heads, mlp_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) stem features
        Returns:
            outputs_map: list of 4 tensors, each (B, H, W, C)
        """
        outputs_map = []
        for block in self.vit_blocks:
            x = block(x)
            outputs_map.append(x)
        return outputs_map


class ViTMoE(nn.Module):
    """
    Vision Transformer with Mixture of Experts.
    Replaces the original ViT classifier with a MoE architecture where
    each expert = a full set of ViTBlocks, and a learned router
    selects which expert(s) to activate per input.
    
    num_experts is fixed at total class count (e.g., 12 for VisA).
    """
    def __init__(self, inplanes=3, hidden_dim=256, num_heads=4, mlp_dim=128,
                 num_experts=12, top_k=1):
        super(ViTMoE, self).__init__()
        self.patch_size = 1
        self.num_patches = inplanes * 14 * 14
        self.embedding_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Shared stem (same as original ViT)
        self.conv1 = nn.Conv2d(inplanes, hidden_dim, kernel_size=self.patch_size,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Router: takes GAP of stem features
        self.router = MoERouter(hidden_dim, num_experts, top_k)

        # Expert pool: each expert is a full set of 4 ViTBlocks
        self.experts = nn.ModuleList([
            Expert(hidden_dim, self.num_patches, num_heads, mlp_dim, num_blocks=4)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: dict with key "image" -> (B, C, H, W)
        Returns:
            dict with:
                "outputs_map": list of 4 tensors (B, H, W, C), detached
                "router_probs": (B, num_experts) full softmax probs
                "expert_indices": (B, top_k) selected expert indices
        """
        # Shared stem
        stem_out = self.conv1(x["image"])   # (B, hidden_dim, H, W)
        stem_out = self.bn1(stem_out)
        stem_out = self.relu(stem_out)

        # Router input: global average pool of stem
        router_input = stem_out.mean(dim=[2, 3])  # (B, hidden_dim)
        top_k_probs, top_k_indices, all_probs = self.router(router_input)
        # top_k_probs:   (B, top_k)
        # top_k_indices:  (B, top_k)
        # all_probs:      (B, num_experts)

        B = stem_out.size(0)
        stem_rearranged = rearrange(stem_out, 'b c h w -> b h w c')  # (B, H, W, C)

        # Initialize combined output: list of 4 layers, each (B, H, W, C)
        combined_outputs = [
            torch.zeros_like(stem_rearranged) for _ in range(4)
        ]

        # Group samples by expert to perform batched forward passes (prevents OOM)
        expert_to_batch = {i: [] for i in range(self.num_experts)}
        expert_to_b_indices = {i: [] for i in range(self.num_experts)}
        expert_to_weights = {i: [] for i in range(self.num_experts)}

        for k_idx in range(self.top_k):
            for b in range(B):
                expert_idx = top_k_indices[b, k_idx].item()
                weight = top_k_probs[b, k_idx]
                
                expert_to_batch[expert_idx].append(stem_rearranged[b])
                expert_to_b_indices[expert_idx].append(b)
                expert_to_weights[expert_idx].append(weight)

        for expert_idx in range(self.num_experts):
            if len(expert_to_batch[expert_idx]) > 0:
                expert_input = torch.stack(expert_to_batch[expert_idx], dim=0)
                expert_weights = torch.stack(expert_to_weights[expert_idx], dim=0).view(-1, 1, 1, 1)
                b_indices = torch.tensor(expert_to_b_indices[expert_idx], device=stem_out.device)
                
                expert_out = self.experts[expert_idx](expert_input)
                
                for layer_i in range(4):
                    weighted_out = expert_out[layer_i] * expert_weights
                    # index_add_ is autograd-friendly and safer than += over non-contiguous loops
                    combined_outputs[layer_i].index_add_(0, b_indices, weighted_out)
                    del weighted_out
                    
                del expert_input, expert_weights, b_indices, expert_out

        # Detach for reconstruction conditioning (same as original)
        outputs_map = [out.detach() for out in combined_outputs]

        return {
            "outputs_map": outputs_map,
            "router_probs": all_probs,        # (B, num_experts) - for locality loss
            "expert_indices": top_k_indices,   # (B, top_k) - for logging/analysis
        }


# Keep backward compatibility alias
ViT = ViTMoE