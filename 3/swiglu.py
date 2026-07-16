import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # Projects input through w1 to get activation component 'a'.
        a = self.w1(x)
        # self.act (nn.SiLU) computes SiLU/Swish activation: b = x_proj * sigmoid(x_proj) = x_proj / (1 + e^-x_proj)
        b = self.act(self.w2(x))
        # Performs element-wise multiplication (a * b) to implement Gated Linear Unit,
        # then projects back using linear layer w3 and applies dropout.
        return self.drop(self.w3(a * b))