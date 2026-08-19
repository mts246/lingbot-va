# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Convert a Frank3 (ROS2 MCAP) manifest into a LeRobot v2.1 dataset for lingbot-va.

Source  : /ytech_milm/collect_data_103/frank3/.../manifests/train_passed.yaml
          (Franka FR3 7-DoF joints + Robotiq gripper + EE pose + 2 cameras, MCAP bags)
Target  : LeRobot v2.1 dataset (meta/*, data/chunk-XXX/*.parquet, videos/chunk-XXX/<cam>/*.mp4)

State/action representation (per user choice):
  observation.state[t] = [ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper.pos]  (8-dim)
  action[t]            = observation.state[t+1]  (shift-by-1; last frame repeats)  by default,
                         or = observation.state[t] when --action_mode same.

Camera mapping (per user choice):
  cam1 -> observation.images.front
  cam2 -> observation.images.wrist.left

Prerequisites (on the conversion machine):
  pip install mcap mcap-ros2-support imageio imageio-ffmpeg pyav pyarrow pandas pyyaml tqdm

Example:
  python script/convert_frank3_to_lerobot.py \
    --manifest /ytech_milm/collect_data_103/frank3/20260714/DATA/manifests/train_passed.yaml \
    --out_dir  /m2v_intern/genghaotian/lingbot-va/data/lerobot_frank3_v21 \
    --task "Pick up the object and place it on the plate" \
    --fps 15

After this, run the latent-encoding preprocess step:
  python script/preprocess_so_arm101.py \
    --dataset_path /m2v_intern/genghaotian/lingbot-va/data/lerobot_frank3_v21 \
    --pretrained   /m2v_intern2/genghaotian/models/Robbyant/lingbot-va-base \
    --config <your_frank3_train_config> --target_fps 15
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

STATE_NAMES = [
    "ee_x", "ee_y", "ee_z",
    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
    "gripper.pos",
]
STATE_DIM = len(STATE_NAMES)

CAM_MAP = {
    "cam1": "observation.images.front",
    "cam2": "observation.images.wrist.left",
}

CHUNKS_SIZE = 1000


FALLBACK_READER_DIRS = [
    "/ytech_milm/collect_data_103/frank3/20260714/DATA/tools",
    "/ytech_milm/collect_data_103/frank3/pipeline",
]


def _find_reader_dir(manifest_path: Path, explicit: str = "") -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    # tools/ next to the manifest's data_root (manifest is under <root>/manifests/)
    candidates.append(manifest_path.parent.parent / "tools")
    candidates.extend(Path(d) for d in FALLBACK_READER_DIRS)
    for d in candidates:
        if (d / "frank3_dataset.py").exists():
            return d
    raise FileNotFoundError(
        "frank3_dataset.py not found. Searched: "
        + ", ".join(str(c) for c in candidates)
        + ". Pass --reader /path/to/dir_containing_frank3_dataset.py")


def load_frank3_reader(manifest_path: Path, reader_dir: str = ""):
    """Import Frank3EpisodeDataset; reuse the copy shipped with any frank3 dataset."""
    tools_dir = _find_reader_dir(manifest_path, reader_dir)
    print(f"[convert] using reader: {tools_dir / 'frank3_dataset.py'}")
    sys.path.insert(0, str(tools_dir.parent))
    sys.path.insert(0, str(tools_dir))
    from frank3_dataset import Frank3EpisodeDataset  # noqa: E402
    return Frank3EpisodeDataset.from_manifest(manifest_path)


def build_state(sample) -> np.ndarray:
    """[T, 8] = ee_pose(7) + gripper(1)."""
    ee = np.asarray(sample["ee_pose"], dtype=np.float32)      # [T,7]
    grip = np.asarray(sample["gripper"], dtype=np.float32)    # [T,1]
    T = min(ee.shape[0], grip.shape[0])
    state = np.concatenate([ee[:T], grip[:T]], axis=1)        # [T,8]
    return state.astype(np.float32)


def build_action(state: np.ndarray, mode: str) -> np.ndarray:
    if mode == "same":
        return state.copy()
    # "next": action[t] = state[t+1]; last frame repeats the final state.
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


def compute_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def compute_image_stats(imgs: np.ndarray) -> dict:
    """imgs uint8 [T,H,W,3] -> per-channel stats in [0,1], shape [3,1,1]."""
    x = imgs.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    mn = x.min(axis=(0, 1, 2))
    mx = x.max(axis=(0, 1, 2))

    def as_chw(v):
        return [[[float(c)]] for c in v]

    return {
        "min": as_chw(mn),
        "max": as_chw(mx),
        "mean": as_chw(mean),
        "std": as_chw(std),
        "count": [int(imgs.shape[0])],
    }


def write_video(frames: np.ndarray, out_path: Path, fps: int, codec: str) -> None:
    """frames uint8 [T,H,W,3] RGB -> mp4."""
    import imageio.v2 as imageio
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec=codec,
        macro_block_size=None, pixelformat="yuv420p",
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def write_parquet(out_path: Path, state, action, episode_index, global_index0,
                  task_index, fps):
    import pandas as pd
    T = state.shape[0]
    ts = (np.arange(T, dtype=np.float32) / float(fps))
    df = pd.DataFrame({
        "action": list(action.astype(np.float32)),
        "observation.state": list(state.astype(np.float32)),
        "timestamp": ts.astype(np.float32),
        "frame_index": np.arange(T, dtype=np.int64),
        "episode_index": np.full(T, episode_index, dtype=np.int64),
        "index": np.arange(global_index0, global_index0 + T, dtype=np.int64),
        "task_index": np.full(T, task_index, dtype=np.int64),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def build_info(total_episodes, total_frames, total_videos, total_chunks,
               fps, codec) -> dict:
    def video_feat():
        return {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
                "video.video_backend": "pyav",
            },
        }

    features = {
        "action": {"dtype": "float32", "names": STATE_NAMES, "shape": [STATE_DIM]},
        "observation.state": {"dtype": "float32", "names": STATE_NAMES, "shape": [STATE_DIM]},
    }
    for cam_key in CAM_MAP.values():
        features[cam_key] = video_feat()
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    return {
        "codebase_version": "v2.1",
        "robot_type": "franka_fr3",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to train_passed.yaml")
    ap.add_argument("--out_dir", required=True, help="output LeRobot v2.1 dataset dir")
    ap.add_argument("--task", required=True, help="language instruction for all episodes")
    ap.add_argument("--reader", default="",
                    help="dir containing frank3_dataset.py (auto-detected if omitted)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--action_mode", choices=["next", "same"], default="next",
                    help="'next': action[t]=state[t+1]; 'same': action[t]=state[t]")
    ap.add_argument("--video_codec", default="libx264",
                    help="ffmpeg codec for videos (e.g. libx264, libsvtav1)")
    ap.add_argument("--info_codec", default="h264",
                    help="value written to info.json video.codec (h264/av1)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only convert first N episodes (debug)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    assert manifest_path.exists(), manifest_path

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    ds = load_frank3_reader(manifest_path, args.reader)
    n = len(ds)
    if args.limit > 0:
        n = min(n, args.limit)
    print(f"[convert] {n} episodes -> {out_dir}")

    episodes_rows = []
    episodes_stats_rows = []
    total_frames = 0
    total_videos = 0
    global_index = 0
    kept = 0

    for ep_idx in tqdm(range(n), desc="episodes"):
        sample = ds[ep_idx]
        T = int(sample["T"])
        if T < 2:
            print(f"  skip ep{ep_idx}: too short (T={T})")
            continue

        state = build_state(sample)          # [T,8]
        T = state.shape[0]
        action = build_action(state, args.action_mode)

        chunk_id = kept // CHUNKS_SIZE

        # parquet
        parquet_path = out_dir / f"data/chunk-{chunk_id:03d}/episode_{kept:06d}.parquet"
        write_parquet(parquet_path, state, action, kept, global_index,
                      task_index=0, fps=args.fps)

        # videos
        stats = {
            "action": compute_stats(action),
            "observation.state": compute_stats(state),
        }
        for cam_src, cam_key in CAM_MAP.items():
            frames = np.asarray(sample["images"][cam_src])  # [T,H,W,3] uint8
            frames = frames[:T]
            vid_path = out_dir / f"videos/chunk-{chunk_id:03d}/{cam_key}/episode_{kept:06d}.mp4"
            write_video(frames, vid_path, args.fps, args.video_codec)
            stats[cam_key] = compute_image_stats(frames)
            total_videos += 1

        episodes_rows.append({
            "episode_index": kept,
            "tasks": [args.task],
            "length": T,
            "action_config": [{
                "start_frame": 0,
                "end_frame": T,
                "action_text": args.task,
                "skill": "",
            }],
        })
        episodes_stats_rows.append({"episode_index": kept, "stats": stats})

        total_frames += T
        global_index += T
        kept += 1

    total_chunks = (kept + CHUNKS_SIZE - 1) // CHUNKS_SIZE if kept else 0

    # meta/tasks.jsonl
    with (out_dir / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}, ensure_ascii=False) + "\n")

    # meta/episodes.jsonl
    with (out_dir / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # meta/episodes_stats.jsonl
    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in episodes_stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # meta/info.json
    info = build_info(kept, total_frames, total_videos, total_chunks,
                      args.fps, args.info_codec)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"[convert] done. episodes={kept} frames={total_frames} videos={total_videos}")
    print(f"[convert] dataset at: {out_dir}")
    print("[convert] next: run script/preprocess_so_arm101.py to encode latents.")


if __name__ == "__main__":
    main()
