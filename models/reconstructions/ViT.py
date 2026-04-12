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
        # Routers: one MoERouter per ViTBlock layer (4 layers)
        self.routers = nn.ModuleList([
            MoERouter(hidden_dim, num_experts, top_k) for _ in range(4)
        ])

        # Expert pool: each expert is a full set of 4 ViTBlocks
        self.experts = nn.ModuleList([
            Expert(hidden_dim, self.num_patches, num_heads, mlp_dim, num_blocks=4)
            for _ in range(num_experts)
        ])

    def forward(self, stem_out):
        """
        Args:
            stem_out: (B, hidden_dim, H, W) features from backbone/projection
        Returns:
            dict with:
                "outputs_map": list of 4 tensors (B, H, W, C), detached
                "router_probs": (B*4, num_experts) full softmax probs across all layers
                "expert_indices": (B*4, top_k) selected expert indices across all layers
        """
        B = stem_out.size(0)
        stem_rearranged = rearrange(stem_out, 'b c h w -> b h w c')  # (B, H, W, C)
        
        x = stem_rearranged
        outputs_map = []
        all_probs_list = []
        top_k_indices_list = []

        for layer_i in range(4):
            # Router input: global average pool of current spatial features
            router_input = x.mean(dim=[1, 2])  # (B, hidden_dim)
            top_k_probs, top_k_indices, all_probs = self.routers[layer_i](router_input)
            
            all_probs_list.append(all_probs)
            top_k_indices_list.append(top_k_indices)
            
            combined_output = torch.zeros_like(x)

            # Group samples by expert to perform batched forward passes (prevents OOM)
            expert_to_batch = {i: [] for i in range(self.num_experts)}
            expert_to_b_indices = {i: [] for i in range(self.num_experts)}
            expert_to_weights = {i: [] for i in range(self.num_experts)}

            for k_idx in range(self.top_k):
                for b in range(B):
                    expert_idx = top_k_indices[b, k_idx].item()
                    weight = top_k_probs[b, k_idx]
                    
                    expert_to_batch[expert_idx].append(x[b])
                    expert_to_b_indices[expert_idx].append(b)
                    expert_to_weights[expert_idx].append(weight)

            for expert_idx in range(self.num_experts):
                if len(expert_to_batch[expert_idx]) > 0:
                    expert_input = torch.stack(expert_to_batch[expert_idx], dim=0)
                    expert_weights = torch.stack(expert_to_weights[expert_idx], dim=0).view(-1, 1, 1, 1)
                    b_indices = torch.tensor(expert_to_b_indices[expert_idx], device=stem_out.device)
                    
                    # Apply ONLY the block `layer_i` from the selected `expert_idx`
                    expert_block = self.experts[expert_idx].vit_blocks[layer_i]
                    expert_out = expert_block(expert_input)
                    
                    weighted_out = expert_out * expert_weights
                    combined_output.index_add_(0, b_indices, weighted_out)
                    
            # Store the output for this layer, detached just like the original logic
            outputs_map.append(combined_output.detach())
            x = combined_output  # Next layer receives combined output

        # Concatenate routing statistics from all 4 layers
        cat_probs = torch.cat(all_probs_list, dim=0)       # (B*4, num_experts)
        cat_indices = torch.cat(top_k_indices_list, dim=0) # (B*4, top_k)

        return {
            "outputs_map": outputs_map,
            "router_probs": cat_probs,        # for locality and balance loss
            "expert_indices": cat_indices,    # for balance loss
        }


# Keep backward compatibility alias
ViT = ViTMoE