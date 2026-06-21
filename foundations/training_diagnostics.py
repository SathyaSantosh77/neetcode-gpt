import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    m = round(x.mean().item(), 4)
                    s = round(x.std().item(), 4)
                    if x.dim() >= 2:
                        df = round(((x <= 0).all(dim = 0)).float().mean().item(), 4)
                    else:
                        df = round((x <= 0).float().mean().item(), 4)
                    stats.append({'mean' : m, 'std': s, 'dead_fraction': df})
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        o = model(x)
        l = nn.MSELoss()(o, y)
        l.backward()
        stats = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                m = round(grad.mean().item(), 4)
                s = round(grad.std().item(), 4)
                n = round(torch.norm(grad).item(), 4)
                stats.append({'mean': m, 'std': s, 'norm': n})

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in activation_stats:
            if i['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for i in gradient_stats:
            if i['norm'] > 1000:
                return 'exploding_gradients'
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        for i in activation_stats:
            if i['std'] < 0.1:
                return 'vanishing_gradients'
            if i['std'] > 10.0:
                return 'exploding_gradients'
        return 'healthy'
