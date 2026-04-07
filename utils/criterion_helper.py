import torch.nn as nn
import torch

class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


class SVD_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, av0,av1,av2, ratio=0.1):
        
        av0 = av0.mean(dim=2)
        av1 = av1.mean(dim=2)
        av2 = av2.mean(dim=2)
        
        s0 = torch.linalg.svdvals(av0)
        s0 = torch.div(s0, torch.sum(s0))
        cov_loss0 = torch.sum(s0[s0 < ratio/256])
        # print(s0[s0 < ratio/256])
        # # print(s0[-int(ratio*5):-1])
        # # print(s0[4:7])
        # print(s0)
        # return
    
        s1 = torch.linalg.svdvals(av1)
        s1 = torch.div(s1, torch.sum(s1))
        cov_loss1 = torch.sum(s1[s1 < ratio/256])

        s2 = torch.linalg.svdvals(av2)
        s2 = torch.div(s2, torch.sum(s2))
        cov_loss2 = torch.sum(s2[s2 < ratio/256])

        return (cov_loss0 + cov_loss1 + cov_loss2)/3


class LocalityLoss(nn.Module):
    """
    Locality loss from "Theory on Mixture-of-Experts in Continual Learning" 
    (arXiv:2406.16437, Eq. 6).
    
    Prevents expert forgetting by penalizing weight changes of experts,
    weighted by their router assignment probability:
    
        L_loc = sum_m  pi_m(X, Theta) * || w_t^(m) - w_{t-1}^(m) ||_2
    
    For experts m != m_t (not updated by current task), w_t^(m) = w_{t-1}^(m),
    so their contribution is zero. This ensures only the actively-routed 
    expert's weight drift is penalized proportionally to routing confidence.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.prev_expert_params = {}  # snapshot from previous task
    
    def snapshot_experts(self, experts_module):
        """
        Call after training on task t-1 to save expert weights.
        This creates a frozen copy of all expert parameters.
        
        Args:
            experts_module: nn.ModuleList of Expert modules
        """
        self.prev_expert_params = {}
        for i, expert in enumerate(experts_module):
            self.prev_expert_params[i] = {
                name: p.clone().detach()
                for name, p in expert.named_parameters()
            }
    
    def has_snapshot(self):
        """Check if a previous snapshot exists."""
        return len(self.prev_expert_params) > 0
    
    def forward(self, router_probs, experts_module):
        """
        Compute locality loss.
        
        Args:
            router_probs: (B, M) softmax probabilities from router
            experts_module: nn.ModuleList of Expert modules
            
        Returns:
            Scalar loss tensor
        """
        if not self.prev_expert_params:
            return torch.tensor(0.0, device=router_probs.device, requires_grad=False)
        
        # Mean routing probability per expert across the batch
        mean_probs = router_probs.mean(dim=0)  # (M,)
        
        loss = torch.tensor(0.0, device=router_probs.device)
        for m, expert in enumerate(experts_module):
            if m not in self.prev_expert_params:
                continue  # new expert or no previous snapshot
            
            param_diff_sq = torch.tensor(0.0, device=router_probs.device)
            for name, p in expert.named_parameters():
                if name in self.prev_expert_params[m]:
                    prev_p = self.prev_expert_params[m][name].to(p.device)
                    param_diff_sq = param_diff_sq + (p - prev_p).pow(2).sum()
            
            # Weight the L2 norm of parameter drift by routing probability
            loss = loss + mean_probs[m] * torch.sqrt(param_diff_sq + 1e-8)
        
        return loss


class LoadBalanceLoss(nn.Module):
    """
    Standard load-balancing loss for Mixture of Experts.
    
    Prevents expert collapse (all inputs routed to same expert) by
    encouraging uniform routing distribution:
    
        L_bal = N * sum_m (f_m * P_m)
    
    where f_m = fraction of tokens routed to expert m,
          P_m = mean router probability for expert m,
          N = number of experts.
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, router_probs, expert_indices, num_experts):
        """
        Args:
            router_probs: (B, M) softmax probabilities from router
            expert_indices: (B, top_k) selected expert indices
            num_experts: int, total number of experts M
            
        Returns:
            Scalar loss tensor
        """
        B = router_probs.size(0)
        device = router_probs.device
        
        # f_m: fraction of samples routed to each expert
        # Count how many times each expert is selected (across all top-k)
        counts = torch.zeros(num_experts, device=device)
        for k in range(expert_indices.size(1)):
            for idx in expert_indices[:, k]:
                counts[idx.item()] += 1
        f_m = counts / (B * expert_indices.size(1))  # normalize
        
        # P_m: mean router probability per expert
        P_m = router_probs.mean(dim=0)  # (M,)
        
        # Load balance loss: N * sum(f_m * P_m)
        loss = num_experts * (f_m * P_m).sum()
        
        return loss

    
def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict

def build_svd_loss():
    loss = SVD_Loss()
    return loss

def build_locality_loss(weight=1.0):
    return LocalityLoss(weight=weight)

def build_load_balance_loss(weight=0.01):
    return LoadBalanceLoss(weight=weight)

