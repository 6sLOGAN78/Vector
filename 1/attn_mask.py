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
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)
