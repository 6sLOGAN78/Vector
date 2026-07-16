from __future__ import annotations
from pathlib import Path
import torch

class ByteDataset:
    """Holds raw bytes of a text file and yields (x,y) blocks for LM.
    - block_size: sequence length (context window)
    - split: fraction for training (rest is val)
    """
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes()
        # Converts raw python bytes list using torch.tensor to a 1D index tensor of type torch.long.
        data = torch.tensor(list(data), dtype=torch.long)
        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        
        # torch.randint generates a 1D tensor of "batch_size" random integers within 
        # the interval [0, len(buf) - block_size - 1). These serve as starting offsets for each sequence.
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        
        # For each index i, we extract a slice of size self.block_size.
        # torch.stack aggregates these 1D slices into a 2D batch tensor of shape (batch_size, block_size).
        # We shift target y by +1 position relative to inputs x to set up the next-token prediction task.
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        
        # .to(device) registers/transfers the tensors to the specified processor device (CPU or GPU memory).
        return x.to(device), y.to(device)