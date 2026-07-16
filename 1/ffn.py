import torch.nn as nn

class FeedForward(nn.Module):
    """1.5 FFN with expansion factor `mult`.

    Dimensions:
      input:     (B, T, d_model)
      inner:     (B, T, mult*d_model)
      output:    (B, T, d_model)
    """
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # nn.Sequential chains the layers. Linears project representations to/from high dim,
        # and nn.GELU applies Gaussian Error Linear Unit activation mathematically defined as:
        # GELU(x) = x * P(X <= x) = x * Phi(x)
        # yielding smooth non-linear representation mappings.
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Passes the tensor through the sequential linear, activation, and dropout layers.
        return self.net(x)