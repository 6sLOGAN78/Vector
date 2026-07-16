"""1.1 Positional encodings (absolute learned + sinusoidal)."""
import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # nn.Embedding acts as a learnable lookup table maps max_len position indices
        # to dense continuous vectors of dimension d_model.
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        # torch.arange generates a 1D sequence of integers from 0 to T-1 representing token positions.
        pos = torch.arange(T, device=x.device)
        pos_emb = self.emb(pos)  # (T, d_model)
        
        # unsqueeze(0) reshapes pos_emb to (1, T, d_model) so PyTorch automatically broadcasts
        # and adds the sequence of position embeddings to each sequence in the batch.
        return x + pos_emb.unsqueeze(0)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # torch.zeros initializes a static tensor buffer of zeros.
        pe = torch.zeros(max_len, d_model)
        # torch.arange generates position indices, and unsqueeze(1) shapes it to a 2D column tensor (max_len, 1).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Computes sinusoidal frequency dividers: 10000^(2i/d_model).
        # We compute this in log space: exp(i * -log(10000)/d_model) for numerical stability.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Slice assignment: even indices (0::2) receive sine wave encodings,
        # and odd indices (1::2) receive cosine wave encodings.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer saves this as part of state_dict so it is saved with the model,
        # but marks it as non-parameter (no gradients calculated or updated via optimizer).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        # Slices first T position encodings and unsqueezes to (1, T, d_model) to broadcast add over batch.
        return x + self.pe[:T].unsqueeze(0)