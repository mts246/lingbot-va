# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Preprocess a LeRobot v2.1 dataset (so_arm101-style) for lingbot-va training.

Steps:
  A) Add `action_config` field to meta/episodes.jsonl (backup original to episodes_ori.jsonl).
  B) Compute per-channel q01/q99 over all action frames and print Python snippet for config.
  C) Encode each (episode, cam_key) video to Wan2.2 VAE latents, encode prompt with T5,
     and save .pth files under latents/chunk-XXX/<cam_key>/episode_XXXXXX_{s}_{e}.pth.
     Also produce empty_emb.pt at dataset root.
  D) Sanity-check by loading LatentLeRobotDataset and printing one sample shape.

Example:
  python script/preprocess_so_arm101.py \
    --dataset_path /m2v_intern/genghaotian/lingbot-va/data/lerobot_so_arm101_task0_v21 \
    --pretrained   /m2v_intern2/genghaotian/models/Robbyant/lingbot-va-base \
    --target_fps 15
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# allow `import wan_va.*`
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'wan_va'))

from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: E402

from wan_va.modules.utils import (  # noqa: E402
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


# ---------- Step A: action_config ----------
def add_action_config(dataset_path: Path) -> None:
    meta_dir = dataset_path / 'meta'
    eps_file = meta_dir / 'episodes.jsonl'
    bak_file = meta_dir / 'episodes_ori.jsonl'
    assert eps_file.exists(), f'{eps_file} not found'

    rows = []
    needs_patch = False
    with eps_file.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if 'action_config' not in row:
                needs_patch = True
            rows.append(row)

    if not needs_patch:
        print(f'[A] action_config already present in {eps_file}, skip.')
        return

    if not bak_file.exists():
        shutil.copy2(eps_file, bak_file)
        print(f'[A] backup -> {bak_file}')

    with eps_file.open('w') as f:
        for row in rows:
            if 'action_config' not in row:
                length = int(row['length'])
                action_text = row['tasks'][0] if row.get('tasks') else ''
                row['action_config'] = [{
                    'start_frame': 0,
                    'end_frame': length,
                    'action_text': action_text,
                    'skill': '',
                }]
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'[A] patched {len(rows)} episodes in {eps_file}')


# ---------- Step B: action stats ----------
def compute_action_stats(dataset_path: Path, action_dim_used: int) -> None:
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

    print('\n================= [B] action q01/q99 =================')
    print(f'#  total frames: {actions.shape[0]}, used dims: {action_dim_used}')
    print('#  Paste the block below into wan_va/configs/va_so_arm101_cfg.py:norm_stat\n')
    print('va_so_arm101_cfg.norm_stat = {')
    print('    "q01": [')
    for v in q01:
        print(f'        {v:.6f},')
    print('    ] + [0.] * 24,')
    print('    "q99": [')
    for v in q99:
        print(f'        {v:.6f},')
    print('    ] + [0.] * 24,')
    print('}\n========================================================\n')


# ---------- Step C: encode videos & prompts to latents ----------
def _read_video_frames(video_path: Path) -> torch.Tensor:
    """Return uint8 tensor [T, H, W, 3]."""
    # Prefer torchvision (supports av1 via pyav backend), fallback to decord.
    try:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(video_path), pts_unit='sec', output_format='THWC')
        return frames
    except Exception:
        import decord  # type: ignore
        decord.bridge.set_bridge('native')
        vr = decord.VideoReader(str(video_path))
        return torch.from_numpy(vr[:].asnumpy())


def _t5_encode(prompt: str, tokenizer, text_encoder, device, dtype,
               max_sequence_length: int = 512) -> torch.Tensor:
    prompt_in = prompt_clean(prompt)
    text_inputs = tokenizer(
        [prompt_in],
        padding='max_length',
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    enc_device = next(text_encoder.parameters()).device
    with torch.no_grad():
        out = text_encoder(text_input_ids.to(enc_device), mask.to(enc_device)).last_hidden_state
    out = out.detach().to(dtype=dtype, device=device)
    out = [u[:v] for u, v in zip(out, seq_lens)]
    out = torch.stack([
        torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in out
    ], dim=0)
    return out[0]  # [L, D]


def _normalize_latents(mu: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, -1, 1, 1, 1)
    inv_std = inv_std.view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) * inv_std).to(mu)


def _select_frame_ids(length: int, ori_fps: int, target_fps: int,
                      frame_chunk_size: int) -> np.ndarray:
    """Sample frames at target_fps, then trim to (4k+1)."""
    stride = max(1, int(round(ori_fps / target_fps)))
    ids = list(range(0, length, stride))
    # need at least one chunk: 4 * frame_chunk_size + 1 frames after sampling
    min_frames = 4 * frame_chunk_size + 1
    if len(ids) < min_frames:
        # take dense sampling
        ids = list(range(0, length))
        stride = 1
    # trim to 4k+1 (causal VAE)
    n = len(ids)
    n_keep = ((n - 1) // 4) * 4 + 1
    n_keep = max(n_keep, min_frames)
    n_keep = min(n_keep, n)
    # if still smaller than min_frames (very short ep), drop episode
    return np.array(ids[:n_keep], dtype=np.int64)


def encode_latents(dataset_path: Path, pretrained: Path, cfg, target_fps: int,
                   device: torch.device) -> None:
    dtype = cfg.param_dtype
    print(f'[C] loading VAE / tokenizer / text_encoder from {pretrained} ...')
    vae = load_vae(str(pretrained / 'vae'), torch_dtype=dtype, torch_device=device)
    tokenizer = load_tokenizer(str(pretrained / 'tokenizer'))
    # keep text_encoder on CPU to save VRAM; it's only used a handful of times
    text_encoder = load_text_encoder(str(pretrained / 'text_encoder'),
                                     torch_dtype=dtype, torch_device='cpu')
    streaming_vae = WanVAEStreamingWrapper(vae)

    latents_mean = torch.tensor(vae.config.latents_mean)
    latents_std = torch.tensor(vae.config.latents_std)
    inv_std = 1.0 / latents_std

    # empty_emb
    empty_path = dataset_path / 'empty_emb.pt'
    if not empty_path.exists():
        empty_emb = _t5_encode('', tokenizer, text_encoder, device, dtype)
        torch.save(empty_emb.cpu(), empty_path)
        print(f'[C] saved {empty_path} shape={tuple(empty_emb.shape)}')
    else:
        print(f'[C] empty_emb exists: {empty_path}')

    # episodes
    with (dataset_path / 'meta' / 'episodes.jsonl').open('r') as f:
        episodes = [json.loads(l) for l in f if l.strip()]

    info = json.loads((dataset_path / 'meta' / 'info.json').read_text())
    ori_fps = int(info['fps'])
    chunks_size = int(info['chunks_size'])
    video_tmpl = info['video_path']  # videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    cam_keys = cfg.obs_cam_keys
    H, W = cfg.height, cfg.width

    for ep in tqdm(episodes, desc='[C] encoding'):
        ep_idx = int(ep['episode_index'])
        length = int(ep['length'])
        chunk_id = ep_idx // chunks_size
        text = ep['tasks'][0] if ep.get('tasks') else ''

        for ac in ep.get('action_config', [{'start_frame': 0, 'end_frame': length, 'action_text': text}]):
            s, e = int(ac['start_frame']), int(ac['end_frame'])
            seg_text = ac.get('action_text', text) or text

            # text emb (per segment)
            text_emb = _t5_encode(seg_text, tokenizer, text_encoder, device, dtype).cpu()

            for cam in cam_keys:
                out_dir = dataset_path / 'latents' / f'chunk-{chunk_id:03d}' / cam
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f'episode_{ep_idx:06d}_{s}_{e}.pth'
                if out_file.exists():
                    continue

                video_path = dataset_path / video_tmpl.format(
                    episode_chunk=chunk_id, video_key=cam, episode_index=ep_idx)
                assert video_path.exists(), f'missing video: {video_path}'

                frames = _read_video_frames(video_path)  # [T,H,W,3] uint8
                T_full = frames.shape[0]
                seg_end = min(e, T_full)
                seg_frames = frames[s:seg_end]  # [Ts,H,W,3]
                seg_len = seg_frames.shape[0]

                frame_ids = _select_frame_ids(seg_len, ori_fps, target_fps, cfg.frame_chunk_size)
                if frame_ids.size < 4 * cfg.frame_chunk_size + 1:
                    print(f'  skip ep{ep_idx} seg[{s},{e}] cam={cam}: too short ({seg_len})')
                    continue

                sel = seg_frames[frame_ids]  # [Fs,H,W,3]
                video_num_frames = int(sel.shape[0])
                video_height = int(sel.shape[1])
                video_width = int(sel.shape[2])

                # [Fs,H,W,3] -> [1,3,Fs,H',W']
                x = sel.float().permute(3, 0, 1, 2)  # [3,Fs,H,W]
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
                x = x.unsqueeze(0) / 255.0 * 2.0 - 1.0
                x = x.to(device=device, dtype=dtype)

                streaming_vae.clear_cache()
                with torch.no_grad():
                    # Causal temporal chunked encoding to avoid OOM:
                    # first chunk = 1 frame, subsequent chunks = 4 frames each.
                    # Wan VAE has temporal stride 4: (1 + 4k) video frames -> (1 + k) latent frames.
                    Fs = x.shape[2]
                    enc_parts = []
                    enc_parts.append(streaming_vae.encode_chunk(x[:, :, :1]))
                    i = 1
                    while i + 4 <= Fs:
                        enc_parts.append(streaming_vae.encode_chunk(x[:, :, i:i + 4]))
                        i += 4
                    enc = torch.cat(enc_parts, dim=2)
                mu, _logvar = torch.chunk(enc, 2, dim=1)
                mu_norm = _normalize_latents(mu, latents_mean.to(mu.device),
                                             inv_std.to(mu.device))
                # mu_norm: [1, C, Fl, Hl, Wl]
                _, C, Fl, Hl, Wl = mu_norm.shape
                latent_flat = mu_norm[0].permute(1, 2, 3, 0).reshape(Fl * Hl * Wl, C).to(dtype).cpu()

                payload = {
                    'latent': latent_flat,
                    'latent_num_frames': int(Fl),
                    'latent_height': int(Hl),
                    'latent_width': int(Wl),
                    'video_num_frames': video_num_frames,
                    'video_height': video_height,
                    'video_width': video_width,
                    'text_emb': text_emb,
                    'text': seg_text,
                    'frame_ids': frame_ids,  # local indices within segment
                    'start_frame': s,
                    'end_frame': e,
                    'fps': int(target_fps),
                    'ori_fps': int(ori_fps),
                }
                torch.save(payload, out_file)


# ---------- Step D: sanity check ----------
def sanity_check(cfg_name: str) -> None:
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
    cfg = VA_CONFIGS[cfg_name]
    dset = MultiLatentLeRobotDataset(cfg, num_init_worker=4)
    assert len(dset) > 0, 'dataset is empty after preprocessing'
    sample = dset[0]
    print(f'[D] dataset size = {len(dset)}')
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f'[D]   {k}: {tuple(v.shape)} {v.dtype}')
        else:
            print(f'[D]   {k}: {type(v).__name__}')


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--pretrained', required=True,
                    help='path containing vae/, tokenizer/, text_encoder/')
    ap.add_argument('--config', default='so_arm101_train',
                    help='config name in VA_CONFIGS (used for latent encoding & sanity check)')
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
        compute_action_stats(dataset_path, action_dim_used)
    if not args.skip_latents:
        encode_latents(dataset_path, pretrained, cfg, args.target_fps,
                       torch.device(args.device))
    if not args.skip_check:
        sanity_check(args.config)
    print('done.')


if __name__ == '__main__':
    main()
