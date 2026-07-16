from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = ByteDataset(args.data, block_size=args.block_size)
    # torch.load deserializes stored checkpoints from disk, and map_location transfers tensors directly to correct device memory.
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    # load_state_dict copies parameters and buffers from state dict into model variables.
    model.load_state_dict(ckpt['model'])

    # model.eval() sets the model to evaluation mode, disabling dropout and batchnorm-like updates.
    model.eval()
    losses = []
    # torch.no_grad() disables gradient calculation context, reducing memory consumption and speeding up inference.
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device)
            _, loss = model(xb, yb)
            # loss.item() extracts the python scalar value from the single-element PyTorch tensor.
            losses.append(loss.item())
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()