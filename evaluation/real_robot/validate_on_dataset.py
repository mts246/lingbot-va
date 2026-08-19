#!/usr/bin/env python
"""Trajectory-replay validation for a trained LingBot-VA so_arm101 checkpoint.

For each latent-frame chunk we feed the model **ground-truth** past observations
and past actions (teacher forcing), so the metric measures the model's
chunk-level prediction quality on the training distribution — not roll-out
drift.

Outputs (under --output-dir/<episode_XXXXXX>/):

  gt_video.mp4              side-by-side (front | wrist.left) GT frames
  pred_video.mp4            VAE-decoded predicted latents, same layout
  side_by_side.mp4          top: GT, bottom: prediction
  actions.png               6 subplots, predicted vs GT action per joint
  metrics.json              per-joint L1/L2 error + summary
  actions_gt.npy            GT (T, 6) float32
  actions_pred.npy          predicted (T, 6) float32

Example:
  python evaluation/real_robot/validate_on_dataset.py \\
    --config-name so_arm101 \\
    --checkpoint  /m2v_intern/genghaotian/lingbot-va/train_out/checkpoints/checkpoint_step_1000 \\
    --dataset-path /m2v_intern/genghaotian/lingbot-va/data/lerobot_so_arm101_task0_v21 \\
    --episode-index 0 \\
    --output-dir  visualization/validation/step1000_ep0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LINGBOT_ROOT = Path("/m2v_intern/genghaotian/lingbot-va").resolve()
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))
if str(LINGBOT_ROOT / "wan_va") not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT / "wan_va"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_video_frames(mp4_path: Path) -> np.ndarray:
    """Return uint8 [T, H, W, 3]."""
    try:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(mp4_path), pts_unit="sec", output_format="THWC")
        return frames.numpy()
    except Exception:
        import decord
        vr = decord.VideoReader(str(mp4_path))
        return vr[:].asnumpy()


def _select_frame_ids(length: int, ori_fps: int, target_fps: int,
                      frame_chunk_size: int) -> np.ndarray:
    """Same sampling as preprocess_so_arm101._select_frame_ids."""
    stride = max(1, int(round(ori_fps / target_fps)))
    ids = list(range(0, length, stride))
    min_frames = 4 * frame_chunk_size + 1
    if len(ids) < min_frames:
        ids = list(range(0, length))
    n = len(ids)
    n_keep = ((n - 1) // 4) * 4 + 1
    n_keep = max(n_keep, min_frames)
    n_keep = min(n_keep, n)
    return np.array(ids[:n_keep], dtype=np.int64)


def _save_video(frames_uint8: np.ndarray, path: Path, fps: int) -> None:
    """frames_uint8: [T, H, W, 3] uint8."""
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), frames_uint8, fps=fps, codec="h264",
                pixelformat="yuv420p", output_params=["-crf", "20"])


def _plot_actions(gt: np.ndarray, pred: np.ndarray,
                  motor_names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, C = gt.shape
    fig, axes = plt.subplots(C, 1, figsize=(10, 2.0 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i in range(C):
        axes[i].plot(gt[:, i], label="GT", color="C0", linewidth=1.2)
        axes[i].plot(pred[:, i], label="Pred", color="C3", linewidth=1.2, linestyle="--")
        axes[i].set_ylabel(motor_names[i])
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("action step")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target-fps", type=int, default=15,
                    help="Same as preprocess_so_arm101.py --target_fps.")
    ap.add_argument("--num-chunks", type=int, default=None,
                    help="Cap number of latent chunks; default = full episode.")
    ap.add_argument("--dist-port", type=int, default=29511)
    ap.add_argument("--decode-only", action="store_true",
                    help="Internal: skip prediction, only load VAE and decode "
                         "the latents saved by a previous predict phase.")
    ap.add_argument("--predict-only", action="store_true",
                    help="Only run prediction and save the latent cache; skip "
                         "decode. Useful for debugging.")
    args = ap.parse_args()

    out_root = Path(args.output_dir) / f"episode_{args.episode_index:06d}"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / "_predict_cache.pt"

    if args.decode_only:
        _run_decode_phase(args, out_root, cache_path)
        return

    # ---- single-rank dist init (VA_Server needs dist.barrier) ----
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.dist_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from configs import VA_CONFIGS
    from distributed.util import init_distributed
    from wan_va_server import VA_Server
    from einops import rearrange

    init_distributed(1, 0, 0)
    cfg = VA_CONFIGS[args.config_name]
    cfg.wan22_pretrained_model_name_or_path = args.checkpoint
    cfg.save_root = args.output_dir
    cfg.infer_mode = "server"
    cfg.host = "127.0.0.1"
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1

    dataset_path = Path(args.dataset_path)

    # ---- load episode metadata ----
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    ori_fps = int(info["fps"])
    chunks_size = int(info["chunks_size"])
    video_tmpl = info["video_path"]
    action_names = info["features"]["action"]["names"]

    with (dataset_path / "meta" / "episodes.jsonl").open() as f:
        ep = None
        for line in f:
            e = json.loads(line)
            if int(e["episode_index"]) == args.episode_index:
                ep = e
                break
    assert ep is not None, f"episode {args.episode_index} not found"
    length = int(ep["length"])
    task = ep["tasks"][0] if ep.get("tasks") else ""
    ep_idx = args.episode_index
    ep_chunk = ep_idx // chunks_size
    print(f"[episode {ep_idx}] len={length} task={task!r}")

    # ---- load actions ----
    import pyarrow.parquet as pq
    ap_path = dataset_path / f"data/chunk-{ep_chunk:03d}/episode_{ep_idx:06d}.parquet"
    actions_gt_full = np.stack(
        pq.read_table(str(ap_path), columns=["action"]).column("action").to_pylist()
    ).astype(np.float32)  # (T=length, 6)
    n_motors = actions_gt_full.shape[1]
    assert n_motors == len(cfg.used_action_channel_ids), \
        f"parquet action dim {n_motors} != used_action_channel_ids {cfg.used_action_channel_ids}"

    # ---- load videos ----
    cams = list(cfg.obs_cam_keys)
    frames_per_cam = {}
    for cam in cams:
        vpath = dataset_path / video_tmpl.format(
            episode_chunk=ep_chunk, video_key=cam, episode_index=ep_idx)
        assert vpath.exists(), f"missing video: {vpath}"
        arr = _load_video_frames(vpath)  # [T, H, W, 3] uint8
        arr = arr[:length]
        frames_per_cam[cam] = arr
        print(f"  cam {cam}: shape={arr.shape}")

    # ---- select frame ids (mirror preprocessing) ----
    frame_chunk_size = int(cfg.frame_chunk_size)
    ap_frame = int(cfg.action_per_frame)  # =8 for so_arm101 (frame_stride*4)
    stride = max(1, int(round(ori_fps / args.target_fps)))

    frame_ids = _select_frame_ids(length, ori_fps, args.target_fps, frame_chunk_size)
    n_sampled = len(frame_ids)  # 1 + 4k
    latent_frames_total = (n_sampled - 1) // 4 + 1
    print(f"  sampled_frames={n_sampled} latent_frames={latent_frames_total} "
          f"frame_stride={stride}")

    # split into chunks of frame_chunk_size latent frames.
    # Chunk c covers sampled indices [c*Fs : c*Fs + Fs+1] (Fs sampled frames + 1
    # sampled frame from the previous chunk kept as init). More concretely we
    # process (Fc=4) latent frames per infer call: this corresponds to
    #   raw sampled frames = 4*Fc = 16 non-init frames + 1 init frame,
    # but with our streaming setup each chunk after the first receives 4 (=Fc)
    # sampled frames (streaming VAE turns 4->1 latent frame with a 1-frame cache).
    # For validation clarity we process each chunk consuming EXACTLY
    #   [start_sampled_idx, start_sampled_idx + Fc]
    # sampled frames.
    n_chunks = (latent_frames_total - 1) // frame_chunk_size + 1
    if args.num_chunks is not None:
        n_chunks = min(n_chunks, args.num_chunks)
    print(f"  processing {n_chunks} chunks (frame_chunk_size={frame_chunk_size})")

    # ---- build VA_Server ----
    print("Loading VA_Server (VAE / T5 / transformer)...")
    va = VA_Server(cfg)
    va.infer({"reset": True, "prompt": task})
    device = va.device
    dtype = va.dtype

    # ---- KV-cache warmup (mirror so_arm101_server) ----
    # Prime the DiT cond channel by feeding 16 copies of the first frame plus
    # a static state built from the ground-truth action at frame_ids[0].
    # This turns "chunk 0" into a chunk-c>=1 style prediction, avoiding the
    # OOD cond-channel distribution that pushes edge-of-range joints to the
    # normalization center.
    #
    # IMPORTANT: Wan VAE causal encoder requires the FIRST call to have T=1
    # (or 4k+1) frames so its feat_cache is properly initialized. Subsequent
    # calls can use T=4k (=16 here). So we first prime with a single-frame
    # encode (which also sets `init_latent`), then feed 16 copies via
    # compute_kv_cache.
    warmup_frame = {cam: np.ascontiguousarray(frames_per_cam[cam][int(frame_ids[0])])
                    for cam in cams}
    warmup_state_vec = actions_gt_full[int(frame_ids[0])].astype(np.float32)  # (6,)
    warmup_state = np.broadcast_to(
        warmup_state_vec[:, None, None],
        (n_motors, frame_chunk_size, ap_frame),
    ).copy().astype(np.float32)
    print(f"  warmup: prime VAE + KV cache with static init frame; "
          f"state={warmup_state_vec.round(2).tolist()}")
    # step 1: 1-frame encode to prime streaming VAE + set init_latent
    va.init_latent = va._encode_obs({"obs": [warmup_frame]})
    # step 2: write KV cache with 16 static frames (4 latent frames worth)
    va.infer({
        "compute_kv_cache": True,
        "obs": [warmup_frame] * (frame_chunk_size * 4),
        "state": warmup_state,
    })

    latent_preds: list[torch.Tensor] = []
    action_preds: list[np.ndarray] = []

    def frame_dict_at(sampled_idx: int) -> dict:
        raw_idx = int(frame_ids[sampled_idx])
        fd = {}
        for cam in cams:
            fd[cam] = np.ascontiguousarray(frames_per_cam[cam][raw_idx])
        return fd

    # For each latent-frame chunk c:
    #   * c==0:  obs_start = frame_ids[0]  (1 init frame passed to _infer)
    #   * c>=1:  compute_kv_cache with sampled frames
    #                [c*Fc*1 - stride0 ... c*Fc] mapped from sampled indices
    #                using frame_chunk_size raw frames.
    for c in range(n_chunks):
        # sampled_idx range: this chunk consumes
        #   [c*Fc*1 - warmup ... c*Fc] to feed the KV cache streaming VAE
        # For streaming Wan VAE, chunk c produces latent frames [c*Fc, (c+1)*Fc).
        # sampled indices consumed:
        #   chunk 0:  [0]          (init)
        #   chunk 1:  [1..4]       (4 sampled frames -> 1 latent frame each after cache)
        #   chunk 2:  [5..8]
        #   ...
        # Actually per compute_kv_cache we feed `frame_chunk_size` sampled frames
        # so that the streaming VAE outputs `frame_chunk_size` latent frames.
        # Wait: after first chunk with 1 sampled frame, each subsequent
        # streaming call needs 4 raw frames -> 1 latent frame. So to get
        # `frame_chunk_size=4` latent frames we need 4*4=16 sampled frames per
        # kv_cache. Let me recompute: streaming VAE temporal stride = 4, so:
        #   sampled_input_frames = latent_frames * 4
        # after the first chunk (which used 1 sampled frame -> 1 latent frame).
        # For each subsequent chunk of frame_chunk_size=4 latent frames we need
        # 16 sampled frames.
        sampled_needed_per_kv = frame_chunk_size * 4  # 16
        if c == 0:
            obs_first = {"obs": [frame_dict_at(0)]}
            action, latent = va._infer(obs_first, frame_st_id=va.frame_st_id)
        else:
            start_sampled = 1 + (c - 1) * sampled_needed_per_kv
            end_sampled = start_sampled + sampled_needed_per_kv
            end_sampled = min(end_sampled, n_sampled)
            if end_sampled - start_sampled < 4:
                print(f"  chunk {c}: not enough sampled frames "
                      f"({end_sampled - start_sampled}); stopping")
                break
            kv_frames = [frame_dict_at(i) for i in range(start_sampled, end_sampled)]
            # ground-truth action state: use actions from the raw frames covered
            # by the *previous* chunk (which is what the model was conditioned
            # on during training via the actions tensor).
            raw_start = int(frame_ids[start_sampled - 1])
            raw_end = raw_start + frame_chunk_size * ap_frame
            raw_end = min(raw_end, actions_gt_full.shape[0])
            gt_slice = actions_gt_full[raw_start:raw_end]  # (Fc*ap_frame, 6)
            if gt_slice.shape[0] < frame_chunk_size * ap_frame:
                pad = np.zeros((frame_chunk_size * ap_frame - gt_slice.shape[0],
                                gt_slice.shape[1]), dtype=np.float32)
                gt_slice = np.concatenate([gt_slice, pad], axis=0)
            gt_state = gt_slice.reshape(frame_chunk_size, ap_frame, n_motors)
            gt_state = np.transpose(gt_state, (2, 0, 1)).astype(np.float32)  # (C, F, H)
            va.infer({
                "compute_kv_cache": True,
                "obs": kv_frames,
                "state": gt_state,
            })
            # After KV cache advances, predict this chunk from the next sampled
            # frame (used as initial obs for _infer, but frame_st_id != 0 so
            # init_latent path is skipped).
            init_sampled = end_sampled - 1  # last frame just consumed
            action, latent = va._infer(
                {"obs": [frame_dict_at(init_sampled)]},
                frame_st_id=va.frame_st_id,
            )
        # collect
        act_np = np.asarray(action).astype(np.float32)  # (6, F, ap_frame)
        # flatten time
        act_flat = np.transpose(act_np, (1, 2, 0)).reshape(-1, n_motors)
        # NOTE: with KV-cache warmup enabled above, chunk 0 is no longer at
        # position 0 (warmup advanced frame_st_id by 5), so the training-time
        # zero-pad prefix does NOT apply to chunk 0 output anymore. Keep all
        # 32 predicted actions.
        action_preds.append(act_flat)
        latent_preds.append(latent.detach().cpu())
        print(f"  chunk {c}: pred_action first={act_flat[0].round(2).tolist()} "
              f"latent_shape={tuple(latent.shape)}")

    # ---- assemble predictions ----
    pred_actions = np.concatenate(action_preds, axis=0)  # (T_pred, 6)
    # crop GT to same length
    T_pred = pred_actions.shape[0]
    # pred action time index i corresponds to raw frame frame_ids[0] + i
    # actually chunk 0 predicts raw frames [0 .. Fc*ap_frame), chunk 1
    # predicts [Fc*ap_frame .. 2*Fc*ap_frame), etc.
    gt_start = int(frame_ids[0])
    gt_actions = actions_gt_full[gt_start:gt_start + T_pred]
    T_common = min(len(gt_actions), T_pred)
    gt_actions = gt_actions[:T_common]
    pred_actions = pred_actions[:T_common]
    print(f"  action tensors: gt={gt_actions.shape} pred={pred_actions.shape}")

    # ---- save latent + action cache and hand off decode to a fresh process ----
    latent_concat = torch.cat(latent_preds, dim=2).cpu()  # (1, C, Ftot, H, W)
    del latent_preds

    # Assemble GT side-by-side frames now (CPU-only work) so decode subprocess
    # doesn't need to reload the raw videos.
    T_latent = latent_concat.shape[2]
    # 4 sampled frames -> 1 latent frame (chunk 0 uses 1 sampled + gets 1 latent).
    # Reconstruct which sampled frames each latent frame corresponds to.
    # For simplicity we crop GT frames to length matching decoded pixel video
    # in the decode phase (we'll produce (T_latent-1)*4 + 1 pixel frames).
    n_pixel_frames = (T_latent - 1) * 4 + 1
    n_pixel_frames = min(n_pixel_frames, len(frame_ids))
    gt_frame_idx = frame_ids[:n_pixel_frames]
    cache = {
        "latent": latent_concat,
        "pred_actions": pred_actions,
        "gt_actions": gt_actions,
        "gt_frames": {cam: frames_per_cam[cam][gt_frame_idx] for cam in cams},
        "cams": list(cams),
        "action_names": list(action_names),
        "target_fps": int(args.target_fps),
        "checkpoint": str(args.checkpoint),
        "n_pred_chunks": len(action_preds),
        "T_common": int(T_common),
        "episode_index": int(args.episode_index),
    }
    torch.save(cache, cache_path)
    print(f"  saved predict cache -> {cache_path} "
          f"(latent {tuple(latent_concat.shape)})")

    if args.predict_only:
        print("--predict-only set: skipping decode phase")
        return

    # ---- fork a fresh subprocess for VAE decode (frees FSDP-pinned memory) ----
    import subprocess
    print("Launching decode subprocess (fresh Python) ...")
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--decode-only",
        "--config-name", args.config_name,
        "--checkpoint", args.checkpoint,
        "--dataset-path", args.dataset_path,
        "--episode-index", str(args.episode_index),
        "--output-dir", args.output_dir,
        "--target-fps", str(args.target_fps),
    ]
    # Free as much as we can before spawning (child inherits fds but its own
    # CUDA context is fresh).
    del latent_concat, cache
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"[warn] decode subprocess exited with code {ret.returncode}")
        sys.exit(ret.returncode)


def _run_decode_phase(args, out_root, cache_path):
    """Load only the VAE (fresh CUDA context) and decode saved latents."""
    assert cache_path.exists(), f"predict cache missing: {cache_path}"
    print(f"[decode-only] loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    latent_concat: torch.Tensor = cache["latent"]
    pred_actions: np.ndarray = cache["pred_actions"]
    gt_actions: np.ndarray = cache["gt_actions"]
    gt_frames_per_cam: dict = cache["gt_frames"]
    cams: list = cache["cams"]
    action_names: list = cache["action_names"]
    target_fps: int = cache["target_fps"]
    checkpoint: str = cache["checkpoint"]
    T_common: int = cache["T_common"]
    n_pred_chunks: int = cache["n_pred_chunks"]

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Load fresh VAE only (no transformer, no T5, no FSDP).
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    vae_path = os.path.join(checkpoint, "vae")
    print(f"[decode-only] loading VAE from {vae_path}")
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()

    latents_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    video_processor = VideoProcessor(vae_scale_factor=1)

    n_cams = len(cams)
    W_total = latent_concat.shape[-1]
    assert W_total % n_cams == 0, f"latent W={W_total} not divisible by {n_cams}"
    W_per = W_total // n_cams
    Ftot = latent_concat.shape[2]

    import gc
    cam_videos = []
    for k in range(n_cams):
        # Decode the ENTIRE time span at once per camera. Wan VAE is a causal
        # streaming decoder: latent frame 0 -> 1 pixel frame, each subsequent
        # latent frame -> 4 pixel frames (total pixel frames = 1 + (Ftot-1)*4).
        # Do NOT split along time here or the temporal upsampling gets broken
        # at chunk boundaries (each split would restart the cache and drop
        # 3 frames per boundary).
        sub = latent_concat[..., k * W_per:(k + 1) * W_per]
        sub = (sub.to(device, dtype) / inv_std + latents_mean).contiguous()
        with torch.no_grad():
            if hasattr(vae, "clear_cache"):
                vae.clear_cache()
            out = vae.decode(sub, return_dict=False)[0]
        vid = video_processor.postprocess_video(out, output_type="np")
        if isinstance(vid, list):
            vid = vid[0]
        vid = np.asarray(vid)
        if vid.ndim == 5:
            vid = vid[0]  # (T, H, W, 3)
        if vid.dtype != np.uint8:
            vid = np.clip(vid * 255.0, 0, 255).astype(np.uint8)
        print(f"  cam {k} decoded video shape={vid.shape}")
        cam_videos.append(vid)
        del sub, out
        torch.cuda.empty_cache()
        gc.collect()

    video_pred = np.concatenate(cam_videos, axis=2)
    print(f"  pred video: shape={video_pred.shape}")

    # GT frames at same output resolution
    Hout, Wout = video_pred.shape[1], video_pred.shape[2] // n_cams
    from PIL import Image
    T_pred_video = video_pred.shape[0]
    gt_video = []
    for i in range(min(T_pred_video, len(next(iter(gt_frames_per_cam.values()))))):
        panels = []
        for cam in cams:
            img = gt_frames_per_cam[cam][i]
            img_r = np.array(Image.fromarray(img).resize((Wout, Hout),
                                                        resample=Image.BILINEAR))
            panels.append(img_r)
        gt_video.append(np.concatenate(panels, axis=1))
    gt_video = np.stack(gt_video, axis=0)

    T_min = min(gt_video.shape[0], video_pred.shape[0])
    side = np.concatenate([gt_video[:T_min], video_pred[:T_min]], axis=1)

    _save_video(video_pred, out_root / "pred_video.mp4", fps=target_fps)
    _save_video(gt_video,   out_root / "gt_video.mp4",   fps=target_fps)
    _save_video(side,       out_root / "side_by_side.mp4", fps=target_fps)
    np.save(out_root / "actions_gt.npy",   gt_actions)
    np.save(out_root / "actions_pred.npy", pred_actions)
    _plot_actions(gt_actions, pred_actions, action_names,
                  out_root / "actions.png")

    l1 = np.abs(pred_actions - gt_actions).mean(axis=0).tolist()
    l2 = np.sqrt(((pred_actions - gt_actions) ** 2).mean(axis=0)).tolist()
    metrics = {
        "episode_index": cache["episode_index"],
        "num_chunks": n_pred_chunks,
        "action_steps_compared": T_common,
        "per_joint_L1": {n: round(v, 4) for n, v in zip(action_names, l1)},
        "per_joint_L2": {n: round(v, 4) for n, v in zip(action_names, l2)},
        "overall_L1": round(float(np.mean(l1)), 4),
        "overall_L2": round(float(np.mean(l2)), 4),
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
