"""Offline decode dumped video latents from a lingbot-va rollout into mp4.

The real-robot server writes one `latents_{frame_st_id}.pt` per inference
chunk under `{save_root}/real/{exp_name}/` (see wan_va_server.py:568). This
script loads the shared Wan VAE, walks those files in chronological order,
decodes each into RGB frames, and writes:

  * one `chunk_{frame_st_id:06d}.mp4` per chunk
  * (optional) one `full_rollout.mp4` concatenating all chunks

Run it AFTER the rollout finishes (or on a separate GPU) — VAE decode is
memory-hungry and would OOM if inlined into the control loop.

Usage:
    python evaluation/real_robot/decode_dumped_latents.py \\
        --latent-dir visualization/real_robot/real/<exp_name> \\
        --out-dir    visualization/real_robot/decoded/<exp_name> \\
        --config-name so_arm101 \\
        --fps 15 --concat

Locate <exp_name>:
    ls visualization/real_robot/real/
    # entries look like "<PROMPT>_YYYYMMDD_HHMMSS"; pick the newest that
    # matches your session's launch time and prompt.
"""
import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repo root on sys.path so we can import wan_va.*
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from diffusers.utils import export_to_video  # noqa: E402
from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.modules.utils import load_vae  # noqa: E402


LOG = logging.getLogger("decode_dumped_latents")

_LATENT_RE = re.compile(r"latents_(\d+)\.pt$")
_DENOISE_RE = re.compile(r"latents_(\d+)_step(\d+)(?:_t(\d+))?\.pt$")


def list_latent_files(latent_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for p in latent_dir.iterdir():
        m = _LATENT_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def list_denoise_files(
    latent_dir: Path,
) -> dict[int, list[tuple[int, int | None, Path]]]:
    """Group latents_{fid}_step{NN}[_t{TT}].pt -> {fid: [(step, t, path)]}."""
    chunks: dict[int, list[tuple[int, int | None, Path]]] = {}
    for p in latent_dir.iterdir():
        m = _DENOISE_RE.match(p.name)
        if m:
            fid, step = int(m.group(1)), int(m.group(2))
            t = int(m.group(3)) if m.group(3) is not None else None
            chunks.setdefault(fid, []).append((step, t, p))
    for fid in chunks:
        chunks[fid].sort(key=lambda x: x[0])
    return chunks


@torch.no_grad()
def decode_latent(vae, latents: torch.Tensor) -> np.ndarray:
    """Mirror wan_va_server.decode_one_video: denorm then decode -> [-1,1]."""
    latents = latents.to(vae.device, vae.dtype)
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]  # [B, C, F, H, W] in [-1,1]
    video = video.clamp(-1, 1).float().cpu().numpy()
    # [B,C,F,H,W] -> [F,H,W,C], take first batch
    video = video[0].transpose(1, 2, 3, 0)  # [F, H, W, C]
    # diffusers.export_to_video (imageio backend, RGB) does `frame * 255` on
    # np.ndarray inputs, so we must hand it float32 in [0, 1] — NOT uint8.
    video = ((video + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)
    return video  # [F, H, W, 3], float32, RGB, range [0,1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-dir", required=True,
                    help="Dir with latents_*.pt (usually "
                         "visualization/real_robot/real/<exp_name>).")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir for chunk_*.mp4 (and full_rollout.mp4 "
                         "if --concat).")
    ap.add_argument("--config-name", required=True,
                    help="Same config-name you launched the server with, "
                         "e.g. so_arm101 or so_arm101_genghaotian. Used only "
                         "to find the VAE checkpoint path.")
    ap.add_argument("--checkpoint", default=None,
                    help="Override wan22_pretrained_model_name_or_path if the "
                         "server used a custom base.")
    ap.add_argument("--fps", type=int, default=15,
                    help="Output video fps. Latents were sampled at "
                         "target_fps (usually 15).")
    ap.add_argument("--device", default="cuda",
                    help="Device for VAE decode. Use 'cpu' if GPU is busy.")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--concat", action="store_true",
                    help="Also write full_rollout.mp4 concatenating all "
                         "decoded chunks in order.")
    ap.add_argument("--denoise", action="store_true",
                    help="Decode per-step denoising latents "
                         "(latents_{fid}_step{NN}.pt, dumped by the server "
                         "with DUMP_DENOISE=1). Writes one "
                         "chunk_{fid}_denoise.mp4 per chunk showing the "
                         "trajectory from noise to clean (one frame per step). "
                         "Combine with --concat to ALSO decode the clean "
                         "latents into chunk_*.mp4 + full_rollout.mp4.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only decode the first N chunks (debug).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    latent_dir = Path(args.latent_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_latent_files(latent_dir)
    if not files and not args.denoise:
        raise SystemExit(f"No latents_*.pt found in {latent_dir}")
    if args.limit:
        files = files[: args.limit]
    LOG.info("Found %d latent files under %s", len(files), latent_dir)

    # Locate VAE checkpoint via the same config the server used.
    config = VA_CONFIGS[args.config_name]
    ckpt_root = args.checkpoint or config.wan22_pretrained_model_name_or_path
    vae_path = os.path.join(ckpt_root, "vae")
    LOG.info("Loading VAE from %s", vae_path)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    vae = load_vae(vae_path, torch_dtype=dtype_map[args.dtype],
                   torch_device=args.device)
    vae.eval()

    if args.denoise:
        chunks = list_denoise_files(latent_dir)
        if not chunks:
            raise SystemExit(
                f"No latents_*_step*.pt found in {latent_dir}. Did you run "
                f"the server with DUMP_DENOISE=1?")
        fids = sorted(chunks)
        if args.limit:
            fids = fids[: args.limit]
        for fid in fids:
            steps = chunks[fid]
            LOG.info("chunk %d: %d denoise steps", fid, len(steps))
            traj: list[np.ndarray] = []
            img_dir = out_dir / f"chunk_{fid:06d}_steps"
            img_dir.mkdir(parents=True, exist_ok=True)
            for step, t, p in steps:
                latents = torch.load(p, map_location="cpu", weights_only=False)
                frames = decode_latent(vae, latents)  # [F, H, W, 3]
                last = frames[-1]                # last video frame per step
                traj.append(last)
                t_tag = f"t{t:04d}" if t is not None else "tNA"
                png = img_dir / f"step{step:02d}_{t_tag}.png"
                Image.fromarray(
                    (last * 255.0).round().clip(0, 255).astype(np.uint8)
                ).save(png)
                del latents, frames
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            out_mp4 = out_dir / f"chunk_{fid:06d}_denoise.mp4"
            export_to_video(list(np.stack(traj, axis=0)), str(out_mp4),
                            fps=args.fps)
            LOG.info("  -> %s (%d steps) + pngs in %s", out_mp4, len(traj),
                     img_dir)
        LOG.info("Done. Decoded denoise trajectories for %d chunks into %s",
                 len(fids), out_dir)
        if not args.concat:
            return
        LOG.info("--concat given together with --denoise: also decoding "
                 "clean latents for chunk_*.mp4 + full_rollout.mp4")

    all_frames: list[np.ndarray] = []
    for i, (fid, p) in enumerate(files):
        latents = torch.load(p, map_location="cpu", weights_only=False)
        if not torch.is_tensor(latents):
            LOG.warning("%s is not a tensor (got %s), skipping", p, type(latents))
            continue
        LOG.info("[%d/%d] decoding %s shape=%s", i + 1, len(files), p.name,
                 tuple(latents.shape))
        frames = decode_latent(vae, latents)  # [F, H, W, 3]
        out_mp4 = out_dir / f"chunk_{fid:06d}.mp4"
        export_to_video(list(frames), str(out_mp4), fps=args.fps)
        LOG.info("  -> %s (%d frames)", out_mp4, frames.shape[0])
        if args.concat:
            all_frames.append(frames)
        # free between chunks
        del latents, frames
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.concat and all_frames:
        merged = np.concatenate(all_frames, axis=0)
        out_mp4 = out_dir / "full_rollout.mp4"
        export_to_video(list(merged), str(out_mp4), fps=args.fps)
        LOG.info("Wrote merged rollout %s (%d frames)", out_mp4, merged.shape[0])

    LOG.info("Done. Decoded %d chunks into %s", len(files), out_dir)


if __name__ == "__main__":
    main()
