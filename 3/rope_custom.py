from __future__ import annotations
import torch
import math

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin
    
    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos
        # torch.arange generates even column indices. division, power (**), and reciprocal (1.0 /)
        # compute the frequency coefficients inv_freq = 1 / (base ** (2i / d_head)).
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        
        # t is a 1D tensor representing sequence positions [0, max_pos-1].
        t = torch.arange(max_pos, device=self.device).float()
        
        # torch.outer computes the outer product t * inv_freq producing a 2D matrix of angles of shape (max_pos, head_dim/2).
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        
        # torch.cos and torch.sin compute element-wise cosine and sine rotation components from the angles.
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1) % 2 == 0
    # unsqueeze(0) expands shape from (T, D/2) to (1, 1, T, D/2) so it automatically broadcasts over batch B and head H.
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Slices x along the last dimension into even-indexed (::2) and odd-indexed (1::2) coordinates.
    # This separates the feature dimension into D/2 pairs of 2D vectors.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    # Mathematically rotates each 2D vector pair [x1, x2] by angle theta using:
    # [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    
    # torch.empty_like allocates uninitialized memory matching the shape and type of x.
    out = torch.empty_like(x)
    # Assigns the rotated coordinate outputs back into their corresponding even/odd locations.
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out
