from __future__ import annotations
import torch

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - logits: (B, vocab)
    Returns filtered logits with -inf for masked entries.
    """
    B, V = logits.shape
    # clone duplicates the logits tensor to avoid in-place corruption of input activations.
    filtered = logits.clone()

    if top_k is not None and top_k < V:
        # torch.topk extracts values and indices of the top_k largest features along dim=-1.
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        # Slices out the k-th largest value and unsqueezes to shape (B, 1) for broadcasting comparisons.
        kth = topk_vals[:, -1].unsqueeze(-1)
        # Masked replacement: sets all logits smaller than the k-th maximum logit to negative infinity.
        filtered[filtered < kth] = float('-inf')

    if top_p is not None and 0 < top_p < 1.0:
        # torch.sort sorts elements along the last dimension in descending order, returning sorted values and indices.
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        # Computes attention probabilities for cumulative check.
        probs = torch.softmax(sorted_logits, dim=-1)
        # torch.cumsum calculates cumulative sum along dim=-1 (from highest to lowest probability).
        cumsum = torch.cumsum(probs, dim=-1)
        
        # Identify coordinates where index-cumulative probability exceeds nucleus threshold top_p.
        mask = cumsum > top_p
        # Force keeping the single highest probability token (index 0) even if its cumulative sum exceeds top_p.
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        
        # torch.full_like allocates a new tensor of same size/type filled with negative infinity.
        filtered = torch.full_like(filtered, float('-inf'))
        # scatter_ writes/scatters values in sorted_logits back to their original indexes using sorted_idx.
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered