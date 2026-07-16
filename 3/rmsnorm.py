import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.pow(2) element-wise squares the inputs x.
        # .mean(dim=-1, keepdim=True) computes the average of these squared values along the last dimension (dim=-1),
        # retaining the dimensions (keepdim=True) for tensor broadcasting.
        # .add(self.eps).sqrt() adds a small numerical stability term epsilon and takes the square root,
        # computing mathematically: rms = sqrt(mean(x^2) + eps)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # Normalizes input tensor x by dividing it by rms, then scales element-wise by learnable parameter weight.
        return (x / rms) * self.weight