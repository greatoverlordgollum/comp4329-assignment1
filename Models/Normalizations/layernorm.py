import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    LayerNorm for sequence features stored as [B, C, L].

    This normalizes each token position independently across channels (C),
    which matches standard Transformer/QANet behavior.

    y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        # Per-channel affine parameters shared across sequence positions.
        self.weight = nn.Parameter(torch.ones(num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, L]; normalize over C at each (B, L).
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
