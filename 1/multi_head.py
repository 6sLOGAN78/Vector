import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):
    """1.4 Multi-head attention with explicit shape tracing.

    Dimensions (before masking):
      x:      (B, T, d_model)
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      ctx:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # (B,T,d_model)
        B, T, C = x.shape
        # self.qkv Linear layer projects input token representations into combined Q, K, and V space.
        qkv = self.qkv(x)                          # (B,T,3*C)
        
        # view reshapes the 3D tensor to (B, T, 3, n_head, d_head) to isolate Q, K, V and heads.
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)  # (B,T,3,heads,dim)
        if self.trace_shapes:
            print("qkv view:", qkv.shape)
            
        # unbind slices the tensor along dimension 2 into a list of 3 separate tensors (q, k, v).
        q, k, v = qkv.unbind(dim=2)               # each: (B,T,heads,dim)
        
        # transpose(1, 2) swaps sequence length (dim 1) and heads (dim 2) to shape (B, heads, T, d_head),
        # allowing PyTorch block matrix operations to compute all heads in parallel.
        q = q.transpose(1, 2)                      # (B,heads,T,dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        # Scale by 1/sqrt(d_head) to prevent dot product growth that triggers vanishing softmax gradients.
        scale = 1.0 / math.sqrt(self.d_head)
        
        # k.transpose(-2, -1) swaps seq_len and head_dim. torch.matmul multiplies (B, heads, T, d_head) 
        # by (B, heads, d_head, T) to calculate similarity matrix of shape (B, heads, T, T).
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,heads,T,T)
        
        mask = causal_mask(T, device=x.device)
        # masked_fill writes -inf to subsequent tokens ensuring zero probability after softmax.
        attn = attn.masked_fill(mask, float('-inf'))
        
        # F.softmax scales the final dimension to produce normalized attention weights.
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        
        # torch.matmul does a weighted sum over value vectors for each attention head.
        ctx = torch.matmul(w, v)                  # (B,heads,T,dim)
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
            
        # transpose(1, 2) returns scale to (B, T, heads, dim). .contiguous() asserts a flat layout in memory,
        # which is required before .view() collapses heads and head_dim back into C (d_model).
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,d_model)
        
        # Final linear projection projects the concatenated head outputs back into the hidden space.
        out = self.proj(out)
        if self.trace_shapes:
            print("out:", out.shape)
        return out, w