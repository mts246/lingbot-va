# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Preprocess a frank3 LeRobot v2.1 dataset for lingbot-va training.

Thin wrapper around script/preprocess_so_arm101.py: reuses the video->latent and
prompt->T5 encoding logic, but uses the frank3 config (8-dim EE+gripper action) and
prints an 8-dim q01/q99 snippet for wan_va/configs/va_frank3_cfg.py.

Example:
  python script/preprocess_frank3.py \
    --dataset_path /m2v_intern/genghaotian/lingbot-va/data/lerobot_frank3_v21 \
    --pretrained   /m2v_intern2/genghaotian/models/Robbyant/lingbot-va-base \
    --config frank3_train --target_fps 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'wan_va'))
sys.path.insert(0, str(_REPO_ROOT / 'script'))

from preprocess_so_arm101 import (  # noqa: E402
    add_action_config,
    encode_latents,
    sanity_check,
)


def compute_action_stats(dataset_path: Path, action_dim_used: int, action_dim_total: int) -> None:
    import pyarrow.parquet as pq
    parquet_files = sorted((dataset_path / 'data').rglob('episode_*.parquet'))
    assert parquet_files, f'no parquet under {dataset_path/"data"}'

    actions = []
    for p in tqdm(parquet_files, desc='[B] reading parquets'):
        tbl = pq.read_table(p, columns=['action'])
        a = np.stack(tbl.column('action').to_pylist()).astype(np.float32)
        actions.append(a)
    actions = np.concatenate(actions, axis=0)
    assert actions.shape[1] >= action_dim_used, (
        f'parquet action dim {actions.shape[1]} < expected {action_dim_used}')
    actions = actions[:, :action_dim_used]

    q01 = np.quantile(actions, 0.01, axis=0).tolist()
    q99 = np.quantile(actions, 0.99, axis=0).tolist()
    pad = action_dim_total - action_dim_used

    print('\n================= [B] action q01/q99 =================')
    print(f'#  total frames: {actions.shape[0]}, used dims: {action_dim_used}')
    print('#  Paste the block below into wan_va/configs/va_frank3_cfg.py:norm_stat\n')
    print('va_frank3_cfg.norm_stat = {')
    print('    "q01": [')
    for v in q01:
        print(f'        {v:.6f},')
    print(f'    ] + [0.] * {pad},')
    print('    "q99": [')
    for v in q99:
        print(f'        {v:.6f},')
    print(f'    ] + [0.] * {pad},')
    print('}\n========================================================\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--pretrained', required=True,
                    help='path containing vae/, tokenizer/, text_encoder/')
    ap.add_argument('--config', default='frank3_train',
                    help='config name in VA_CONFIGS')
    ap.add_argument('--target_fps', type=int, default=15)
    ap.add_argument('--skip_action_config', action='store_true')
    ap.add_argument('--skip_stats', action='store_true')
    ap.add_argument('--skip_latents', action='store_true')
    ap.add_argument('--skip_check', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    pretrained = Path(args.pretrained)
    assert dataset_path.exists(), dataset_path
    assert pretrained.exists(), pretrained

    from wan_va.configs import VA_CONFIGS
    cfg = VA_CONFIGS[args.config]
    action_dim_used = len(cfg.used_action_channel_ids)

    if not args.skip_action_config:
        add_action_config(dataset_path)
    if not args.skip_stats:
        compute_action_stats(dataset_path, action_dim_used, cfg.action_dim)
    if not args.skip_latents:
        encode_latents(dataset_path, pretrained, cfg, args.target_fps,
                       torch.device(args.device))
    if not args.skip_check:
        sanity_check(args.config)
    print('done.')


if __name__ == '__main__':
    main()
