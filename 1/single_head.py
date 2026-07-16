import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """1.3 Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        super().__init__()
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # x: (B, T, d_model)
        B, T, _ = x.shape
        # nn.Linear projects x to query (q), key (k), and value (v) spaces.
        q = self.q(x)  # (B,T,d_k)
        k = self.k(x)  # (B,T,d_k)
        v = self.v(x)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        
        # Scaling factor 1/sqrt(d_k) keeps the dot products from growing too large in magnitude
        # which would result in extremely small gradients after softmax.
        scale = 1.0 / math.sqrt(q.size(-1))
        
        # k.transpose(-2, -1) swaps the sequence length (T) and feature (d_k) dimensions of keys.
        # torch.matmul performs batch matrix multiplication: Q @ K^T, resulting in shape (B, T, T).
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  
        
        mask = causal_mask(T, device=x.device)
        # squeeze(1) removes the head dimension of size 1 to match the 3D shape (B, T, T) of attn.
        # masked_fill overwrites masked positions (where mask is True) with negative infinity.
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        
        # F.softmax computes standard softmax along the last dimension (dim=-1) to convert scores
        # into a probability distribution over the sequence tokens.
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        
        # torch.matmul does a weighted sum of the value vectors: W @ V, yielding shape (B, T, d_k).
        out = torch.matmul(w, v)  
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w