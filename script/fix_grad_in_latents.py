"""Strip requires_grad / grad_fn from cached latent .pth files in-place."""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm


def detach_tensor(v):
    if torch.is_tensor(v):
        return v.detach().clone().requires_grad_(False)
    return v


def fix_file(p: Path) -> bool:
    obj = torch.load(p, weights_only=False, map_location='cpu')
    changed = False
    if torch.is_tensor(obj):
        if obj.requires_grad or obj.grad_fn is not None:
            obj = detach_tensor(obj)
            changed = True
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            if torch.is_tensor(v) and (v.requires_grad or v.grad_fn is not None):
                obj[k] = detach_tensor(v)
                changed = True
    if changed:
        torch.save(obj, p)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    args = ap.parse_args()
    root = Path(args.dataset_path)

    files = list(root.rglob('empty_emb.pt')) + list((root / 'latents').rglob('*.pth')) if (root / 'latents').exists() else list(root.rglob('empty_emb.pt'))
    n_fixed = 0
    for p in tqdm(files, desc='detaching'):
        if fix_file(p):
            n_fixed += 1
    print(f'fixed {n_fixed} / {len(files)} files')


if __name__ == '__main__':
    main()
