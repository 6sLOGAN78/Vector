from __future__ import annotations
import torch

class ByteTokenizer:
    """Byte-level tokenizer.
    - encode(str) -> LongTensor [N]
    - decode(Tensor[int]) -> str
    - vocab_size = 256
    """
    def encode(self, s: str) -> torch.Tensor:
        # s.encode('utf-8') returns raw UTF-8 bytes.
        # torch.tensor casts the list of byte integers (0-255) to a 1D PyTorch tensor of type torch.long.
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        # Converts PyTorch tensor containing token IDs back to a standard Python list of integers.
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Decode the list of byte integers back to a UTF-8 string, ignoring invalid byte sequences.
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256