"""
UniAD  —  Unified Anomaly Detection reconstruction head.

Uses ViTDecoder (ViT + MoE Adapters) to reconstruct backbone features.
Anomaly score = pixel-wise L2 distance between original and reconstructed features.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.initializer import initialize_from_cfg
from .vis_decoder import ViTDecoder


class UniAD(nn.Module):
    """
    Parameters
    ----------
    inplanes      : list[int]   — [C]  input feature channels
    instrides     : list[int]   — [S]  backbone feature stride (for anomaly map upsampling)
    feature_size  : tuple(H, W) — spatial size of feature map
    feature_jitter: cfg | None  — jitter augmentation config {scale, prob}
    neighbor_mask : (unused, kept for config compatibility)
    hidden_dim    : int         — transformer hidden dim
    pos_embed_type: (unused, kept for config compatibility)
    save_recon    : cfg | None  — {save_dir} to dump reconstructions during eval
    initializer   : cfg | None  — weight initializer
    num_experts   : int         — number of MoE adapter experts per block
    top_k         : int         — top-k routing
    num_layers    : int         — number of transformer blocks  (default 4)
    num_heads     : int         — attention heads               (default 8)
    ffn_num       : int         — adapter bottleneck dim        (default 64)
    task_id       : int         — current task index; -1 = shared router
    apply_moe     : bool        — enable MoE (False = single adapter per block)
    autorouter    : bool        — auto-select router by val_task_id at inference
    """

    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        num_experts: int  = 12,
        top_k:       int  = 2,
        num_layers:  int  = 4,
        num_heads:   int  = 8,
        ffn_num:     int  = 64,
        task_id:     int  = -1,
        apply_moe:   bool = True,
        autorouter:  bool = False,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes,  list) and len(inplanes)  == 1
        assert isinstance(instrides, list) and len(instrides) == 1

        self.feature_size  = feature_size
        self.feature_jitter = feature_jitter
        self.save_recon    = save_recon

        stride = instrides[0]

        # ── ViTDecoder handles: input_proj, positional embed, transformer, output_proj
        self.decoder = ViTDecoder(
            inplanes    = inplanes,
            instrides   = [1],          # spatial upsample is done below (anomaly map only)
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            num_heads   = num_heads,
            num_experts = num_experts,
            topk        = top_k,
            ffn_num     = ffn_num,
            task_id     = task_id,
            apply_moe   = apply_moe,
            autorouter  = autorouter,
            initializer = None,         # handled by outer initialize_from_cfg
        )

        # ── Upsample anomaly score to original image resolution
        self.upsample = (nn.UpsamplingBilinear2d(scale_factor=stride)
                         if stride > 1 else nn.Identity())

        initialize_from_cfg(self, initializer)

    # ── interface methods ─────────────────────────────────────────────────────

    def get_outplanes(self):
        return 272

    def get_outstrides(self):
        return 1

    def get_experts(self):
        """Expose expert modules for LocalityLoss tracking."""
        return self.decoder.get_experts()

    def set_task_id(self, task_id: int):
        """Switch active router/adapter task index at runtime."""
        self.decoder.set_task_id(task_id)

    # ── train/eval propagation ────────────────────────────────────────────────

    def train(self, mode: bool = True):
        self.decoder.train(mode)
        return super().train(mode)

    # ── jitter augmentation ───────────────────────────────────────────────────

    def add_jitter(self, feature_align: torch.Tensor, scale: float, prob: float):
        if random.uniform(0, 1) <= prob:
            B, C, H, W = feature_align.shape
            feature_norms = feature_align.norm(dim=1, keepdim=True) / C  # B x 1 x H x W
            jitter = torch.randn_like(feature_align) * feature_norms * scale
            feature_align = feature_align + jitter
        return feature_align

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, input: dict) -> dict:
        """
        input keys consumed:
            "feature_align" : (B, C, H, W)   backbone features
            "clsname"       : list[str]       class names  (eval + save_recon only)
            "filename"      : list[str]       file paths   (eval + save_recon only)
            "class_out"     : any | None      passed through

        Returns dict:
            "feature_rec"    : (B, C, H, W)
            "feature_align"  : (B, C, H, W)  (may be jittered during training)
            "pred"           : (B, 1, H', W') anomaly score, upsampled
            "moe_loss"       : scalar         load-balance loss from MoE routing
            "class_out"      : passed through
        """
        feature_align = input["feature_align"]          # B x C x H x W

        # Optional feature-space jitter (train-time augmentation)
        if self.training and self.feature_jitter:
            feature_align = self.add_jitter(
                feature_align,
                self.feature_jitter.scale,
                self.feature_jitter.prob,
            )

        # ── ViT decoder forward ───────────────────────────────────────────────
        dec_out     = self.decoder({"feature_align": feature_align})
        feature_rec = dec_out["feature_rec"]    # B x C x H x W
        moe_loss    = dec_out["moe_loss"]       # scalar

        # ── optional reconstruction saving (eval only) ────────────────────────
        if not self.training and self.save_recon:
            clsnames  = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, fname   = os.path.split(filename)
                _, defename      = os.path.split(filedir)
                fname_, _        = os.path.splitext(fname)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, fname_ + ".npy"),
                        feat_rec.detach().cpu().numpy())

        # ── anomaly score: L2 distance ────────────────────────────────────────
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )                                               # B x 1 x H x W
        pred = self.upsample(pred)                      # B x 1 x H' x W'

        return {
            "feature_rec":    feature_rec,
            "feature_align":  feature_align,
            "pred":           pred,
            "moe_loss":       moe_loss,
            "class_out":      input.get("class_out", None),
        }
