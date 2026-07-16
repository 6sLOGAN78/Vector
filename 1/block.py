import torch.nn as nn
from multi_head import MultiHeadSelfAttention
from ffn import FeedForward

class TransformerBlock(nn.Module):
    """1.6 Transformer block = LN → MHA → residual → LN → FFN → residual."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        # nn.LayerNorm performs normalization over the last dimension (d_model) for each token.
        # This standardizes activations, stabilizing network dynamics and improving convergence.
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        # Operations below add a residual connection (identity mapping) to the layer outputs.
        # x + self.attn(...) and x + self.ffn(...) allow gradients to flow unimpeded backward
        # during backpropagation, addressing vanishing gradient issues.
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x