import torch

def causal_mask(size: int, device: torch.device = None):
    """
    Generates a 4D causal mask of shape (1, 1, size, size) representing
    which elements can attend to which. In a causal setting, a query at index i 
    cannot attend to key at index j if j > i.
    
    Returns:
        torch.Tensor: A boolean tensor of shape (1, 1, size, size) where 
                  elements in the upper triangle (excluding diagonal) are True.
    """
    # torch.triu computes the upper triangular part of a matrix, offset by diagonal=1.
    # This identifies elements where column index j is strictly greater than row index i (j > i),
    # representing future token positions in autoregressive attention.
    # Calling .bool() casts the mask to a boolean tensor.
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    
    # unsqueeze adds single-dimensional dimensions at index 0 twice to expand the shape
    # from (size, size) to (1, 1, size, size) for batch-and-head-aware PyTorch broadcasting.
    return mask.unsqueeze(0).unsqueeze(0)

