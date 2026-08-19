#!/usr/bin/env python
"""LingBot-VA async inference server for SO-ARM101 real-robot deployment.

gRPC service compatible with lerobot.async_inference.robot_client (same protocol
as FastWAM's fastwam_async_server). The robot side can keep using the standard
lerobot async client + ws_tcp_tunnel; this server replaces the policy backend
with our trained LingBot-VA so_arm101 model.

Usage on KML (after training finished and attn_mode flipped back to "torch"):

  python evaluation/real_robot/so_arm101_server.py \\
      --config-name so_arm101_genghaotian \\
      --checkpoint  /path/to/runs/<RUN>/checkpoint_step_NNNNN \\
      --prompt      "Place the black bottle cap into the white paper cup" \\
      --port 15173 \\
      --actions-per-chunk 16 --fps 30 \\
      --cam-keys observation.images.front observation.images.wrist.left \\
      --motor-keys shoulder_pan.pos shoulder_lift.pos elbow_flex.pos \\
                    wrist_flex.pos wrist_roll.pos gripper.pos

Then on KML run the WebSocket tunnel server, and on the robot host (4070) run
the standard `python -m lerobot.async_inference.robot_client ...`. See FastWAM
script header for full client-side launch lines.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import threading
import time
import types
from collections import deque
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import torch
import torch.nn.functional as F

# ---- locate sibling repos ----------------------------------------------------
LINGBOT_ROOT = Path("/m2v_intern/genghaotian/lingbot-va").resolve()
LEROBOT_SRC = Path("/m2v_intern/tujiahang/Projects/lerobot/src").resolve()

if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))
if str(LINGBOT_ROOT / "wan_va") not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT / "wan_va"))


def _load_file_as_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for pkg in ("lerobot", "lerobot.transport"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))

services_pb2 = _load_file_as_module(
    "lerobot.transport.services_pb2",
    LEROBOT_SRC / "lerobot/transport/services_pb2.py",
)
services_pb2_grpc = _load_file_as_module(
    "lerobot.transport.services_pb2_grpc",
    LEROBOT_SRC / "lerobot/transport/services_pb2_grpc.py",
)

# transfer-state constants for the chunked observation stream.
_TRANSFER_BEGIN = services_pb2.TransferState.TRANSFER_BEGIN  # type: ignore[attr-defined]
_TRANSFER_MIDDLE = services_pb2.TransferState.TRANSFER_MIDDLE  # type: ignore[attr-defined]
_TRANSFER_END = services_pb2.TransferState.TRANSFER_END  # type: ignore[attr-defined]


def receive_bytes_in_chunks(iterator, queue, shutdown_event):
    import io
    buf = io.BytesIO()
    for item in iterator:
        if shutdown_event.is_set():
            return None
        if item.transfer_state == _TRANSFER_BEGIN:
            buf.seek(0); buf.truncate(0); buf.write(item.data)
        elif item.transfer_state == _TRANSFER_MIDDLE:
            buf.write(item.data)
        elif item.transfer_state == _TRANSFER_END:
            buf.write(item.data); data = buf.getvalue()
            buf.seek(0); buf.truncate(0)
            if queue is not None:
                queue.put(data)
            else:
                return data
        else:
            raise ValueError(f"unknown transfer_state {item.transfer_state}")
    return None


# ---- lerobot helpers stubs (so robot_client's pickled objects deserialize) ---
Action = torch.Tensor
RawObservation = dict[str, Any]


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self): return self.timestamp
    def get_timestep(self): return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action
    def get_action(self): return self.action


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False
    def get_observation(self): return self.observation


_fake_helpers = types.ModuleType("lerobot.async_inference.helpers")
_fake_helpers.TimedData = TimedData
_fake_helpers.TimedAction = TimedAction
_fake_helpers.TimedObservation = TimedObservation
sys.modules.setdefault("lerobot.async_inference",
                       types.ModuleType("lerobot.async_inference"))
sys.modules["lerobot.async_inference.helpers"] = _fake_helpers
for _cls in (TimedData, TimedAction, TimedObservation):
    _cls.__module__ = "lerobot.async_inference.helpers"


# ---- image dump helpers (mirrors FastWAM style) -----------------------------
def _save_uint8_image(img: np.ndarray, path: Path) -> None:
    """Save HxWxC uint8 RGB array to PNG."""
    from PIL import Image
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"unexpected raw image shape {img.shape}")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)


def _save_unit_tensor(t: torch.Tensor, path: Path) -> None:
    """Save a [3,H,W] tensor in [0,1] (pre-normalize) to PNG."""
    from PIL import Image
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(t.shape)}")
    arr = t.detach().float().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_normalized_tensor(t: torch.Tensor, path: Path) -> None:
    """Save a [3,H,W] tensor in [-1,1] back to PNG (un-normalize first)."""
    from PIL import Image
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(t.shape)}")
    arr = (t.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1).mul(255).byte()
    arr = arr.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _cfg_to_dict(cfg) -> dict:
    """EasyDict -> JSON-serializable dict (drops non-serializable like torch.dtype)."""
    out = {}
    for k, v in dict(cfg).items():
        if isinstance(v, (str, int, float, bool, list, tuple)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {kk: list(vv) if hasattr(vv, "__iter__") and not isinstance(vv, str)
                      else vv for kk, vv in v.items()}
        else:
            out[k] = str(v)
    return out


# ---- LingBot-VA imports (done after sys.path setup) -------------------------
# We init a 1-process distributed group so VA_Server's `dist.barrier()` works.
import torch.distributed as dist  # noqa: E402

from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402
from wan_va_server import VA_Server  # noqa: E402

LOG = logging.getLogger("so_arm101_server")


# ---- servicer ----------------------------------------------------------------
class LingBotVAServicer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.shutdown_event = threading.Event()
        self.lock = threading.RLock()
        self.last_obs: TimedObservation | None = None
        self.last_action_chunk: torch.Tensor | None = None  # last predicted [C,F,H]
        self.has_kv_cached = False
        self.warmup_done = bool(getattr(args, "no_warmup", False))
        if self.warmup_done:
            LOG.info("--no-warmup set: skipping KV-cache warmup pass.")
        # Sliding buffer of frame_dicts derived from SendObservations stream,
        # used to feed the streaming VAE with frame_chunk_size frames per
        # compute_kv_cache call (matches the robotwin eval-client behavior
        # of collecting key frames during chunk execution).
        self.obs_buffer: deque = deque(maxlen=128)

        # ---- build lingbot-va config ----
        LOG.info("Loading lingbot-va config %r", args.config_name)
        config = VA_CONFIGS[args.config_name]
        if args.checkpoint is not None:
            config.wan22_pretrained_model_name_or_path = args.checkpoint
            LOG.info("Override wan22_pretrained_model_name_or_path -> %s",
                     args.checkpoint)
        save_root = args.save_root or "visualization/real_robot"
        os.makedirs(save_root, exist_ok=True)
        config.save_root = save_root
        if args.video_steps is not None:
            LOG.info("Override num_inference_steps %s -> %d",
                     getattr(config, "num_inference_steps", None), args.video_steps)
            config.num_inference_steps = args.video_steps
        if args.action_steps is not None:
            LOG.info("Override action_num_inference_steps %s -> %d",
                     getattr(config, "action_num_inference_steps", None), args.action_steps)
            config.action_num_inference_steps = args.action_steps
        if args.video_exec_step is not None:
            LOG.info("Override video_exec_step %s -> %d",
                     getattr(config, "video_exec_step", None), args.video_exec_step)
            config.video_exec_step = args.video_exec_step
        config.host = "0.0.0.0"
        config.infer_mode = "server"
        # singleton distributed (rank=0, world=1) so VA_Server can call dist ops
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.dist_port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed(1, 0, 0)
        config.rank = 0
        config.local_rank = 0
        config.world_size = 1

        self.config = config
        self.cam_keys = list(args.cam_keys) if args.cam_keys else list(config.obs_cam_keys)
        self.motor_keys = list(args.motor_keys)
        assert len(self.motor_keys) == len(config.used_action_channel_ids), (
            f"motor_keys ({len(self.motor_keys)}) must match "
            f"used_action_channel_ids ({len(config.used_action_channel_ids)})")
        self.n_motors = len(self.motor_keys)
        self.frame_chunk_size = int(config.frame_chunk_size)
        self.action_per_frame = int(config.action_per_frame)
        self.actions_per_chunk = int(args.actions_per_chunk)
        self.fps = int(args.fps)
        self.target_fps = int(args.target_fps)
        self.prompt = args.prompt

        LOG.info("Instantiating VA_Server (this loads VAE / T5 / transformer)...")
        self.va = VA_Server(config)
        LOG.info("Initial reset with prompt: %r", self.prompt)
        self.va.infer({"reset": True, "prompt": self.prompt})

        # ---- dump dir (FastWAM-style) ----
        self.dump_dir_root: Path | None = None
        self.dump_dir: Path | None = None
        self._session_idx = 0
        if args.dump_images_dir:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
            self.dump_dir_root = Path(args.dump_images_dir) / f"run_{run_tag}"
            self.dump_dir_root.mkdir(parents=True, exist_ok=True)
            # 1) server CLI args
            (self.dump_dir_root / "server_args.json").write_text(
                json.dumps(vars(args), indent=2, default=str))
            # 2) full lingbot-va config snapshot
            (self.dump_dir_root / "va_config.json").write_text(
                json.dumps(_cfg_to_dict(config), indent=2, default=str))
            # 3) first session subdir
            self.dump_dir = self.dump_dir_root / f"session_{self._session_idx:03d}"
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            # Cross-link session_000 with the initial latent dump dir set up
            # by wan_va_server._reset() at construction. Subsequent sessions
            # get their own link inside SendPolicyInstructions.
            try:
                latent_dir = Path(self.va.exp_save_root).resolve()
                (self.dump_dir / "latent_dir.txt").write_text(
                    str(latent_dir) + "\n")
                link = self.dump_dir / "latents"
                if not (link.exists() or link.is_symlink()):
                    link.symlink_to(latent_dir)
            except Exception as e:  # noqa: BLE001
                LOG.warning("Could not link initial latent_dir: %s", e)
            LOG.info("Image dump enabled -> %s", self.dump_dir_root)

        LOG.info(
            "Ready. cam_keys=%s motor_keys=%s frame_chunk_size=%d "
            "action_per_frame=%d actions_per_chunk=%d fps=%d",
            self.cam_keys, self.motor_keys, self.frame_chunk_size,
            self.action_per_frame, self.actions_per_chunk, self.fps,
        )

    # ---- RPCs ----
    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        with self.lock:
            self.last_obs = None
            self.last_action_chunk = None
            self.obs_buffer.clear()
            self.has_kv_cached = False
            self.warmup_done = bool(getattr(self.args, "no_warmup", False))
            LOG.info("New session: resetting VA model with prompt=%r", self.prompt)
            self.va.infer({"reset": True, "prompt": self.prompt})
            # rotate dump session dir
            if self.dump_dir_root is not None:
                self._session_idx += 1
                self.dump_dir = self.dump_dir_root / f"session_{self._session_idx:03d}"
                self.dump_dir.mkdir(parents=True, exist_ok=True)
                LOG.info("New dump session -> %s", self.dump_dir)
                # Cross-link this session with the latent dump dir created by
                # wan_va_server._reset (which uses a timestamped exp_name).
                # Both a plain-text pointer and a relative symlink are written
                # so `decode_dumped_latents.py --latent-dir` can find them.
                try:
                    latent_dir = Path(self.va.exp_save_root).resolve()
                    (self.dump_dir / "latent_dir.txt").write_text(
                        str(latent_dir) + "\n")
                    link = self.dump_dir / "latents"
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(latent_dir)
                    LOG.info("  latent dump -> %s", latent_dir)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Could not link latent_dir: %s", e)
                # always persist raw pickle bytes
                raw = bytes(request.data)
                try:
                    (self.dump_dir / "client_policy_setup.pkl").write_bytes(raw)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Failed to write client_policy_setup.pkl: %s", e)
                # best-effort decode -> json
                try:
                    import io
                    import pickle as _pkl

                    class _Stub:
                        def __init__(self, *a, **kw):
                            self.__dict__.update(kw)
                            if a: self.__dict__["_args"] = a
                        def __setstate__(self, state):
                            if isinstance(state, dict): self.__dict__.update(state)
                            else: self.__dict__["_state"] = state

                    class _PermissiveUnpickler(_pkl.Unpickler):
                        def find_class(self, module, name):
                            try:
                                return super().find_class(module, name)
                            except Exception:
                                return type(name, (_Stub,), {"__module__": module})

                    cfg_obj = _PermissiveUnpickler(io.BytesIO(raw)).load()
                    fields = {
                        k: getattr(cfg_obj, k)
                        for k in ("policy_type", "pretrained_name_or_path",
                                  "actions_per_chunk", "device",
                                  "rename_map", "lerobot_features")
                        if hasattr(cfg_obj, k)
                    }
                    (self.dump_dir / "client_policy_setup.json").write_text(
                        json.dumps(fields, indent=2,
                                   default=lambda o: getattr(o, "__dict__", str(o))))
                    LOG.info("Client policy setup: %s", fields)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("Could not decode PolicySetup (.pkl kept): %s", e)
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        try:
            buf = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event)
            if buf is None:
                return services_pb2.Empty()
            obs: TimedObservation = pickle.loads(buf)  # nosec
            with self.lock:
                self.last_obs = obs
                # buffer the frame_dict so we can feed a multi-frame chunk to
                # the streaming VAE on the next compute_kv_cache.
                try:
                    fd = self._build_frame_dict(obs.get_observation())
                    self.obs_buffer.append((obs.get_timestep(), fd))
                except Exception:  # noqa: BLE001
                    LOG.exception("build_frame_dict in SendObservations failed")
            LOG.debug("obs step=%d t=%.3f (buf=%d)", obs.get_timestep(),
                      obs.get_timestamp(), len(self.obs_buffer))
        except grpc.RpcError as e:
            LOG.debug("SendObservations closed by peer: %s", e)
        except Exception:
            LOG.exception("SendObservations failed")
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        # Serialize the entire inference pass. The gRPC server uses a
        # ThreadPoolExecutor(max_workers>=2), so without this lock a client
        # retry (e.g. after a deadline) can enter _predict concurrently with
        # an in-flight call. That produces interleaved video/action denoise
        # loops on the same CUDA context and corrupts DTensor dispatch state
        # (observed as a crash inside _try_replicate_spec_for_scalar_tensor).
        with self.lock:
            obs = self.last_obs
            if obs is None:
                return services_pb2.Actions(data=b"")
            try:
                chunk = self._predict(obs)
                if not chunk:
                    return services_pb2.Actions(data=b"")
                return services_pb2.Actions(data=pickle.dumps(chunk))
            except Exception:
                LOG.exception("GetActions: inference failed")
                return services_pb2.Actions(data=b"")

    # ---- inference -----------------------------------------------------------
    def _build_frame_dict(self, raw_obs: RawObservation) -> dict:
        """Pick the two camera arrays out of the lerobot raw obs dict.

        Returns a dict whose keys are exactly `config.obs_cam_keys` (which the
        VA `_encode_obs` iterates in order), values are HxWx3 uint8 RGB arrays.
        """
        frame = {}
        for ck_robot, ck_va in zip(self.cam_keys, self.config.obs_cam_keys):
            if ck_robot not in raw_obs:
                raise KeyError(
                    f"Camera key {ck_robot!r} missing in robot obs; got "
                    f"keys={list(raw_obs)}")
            img = np.asarray(raw_obs[ck_robot])
            if img.ndim != 3 or img.shape[2] not in (3, 4):
                raise ValueError(f"unexpected image shape {img.shape}")
            if img.shape[2] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            frame[ck_va] = np.ascontiguousarray(img)
        return frame

    def _build_state_tensor(self, raw_obs: RawObservation) -> np.ndarray:
        """[deprecated] previously replicated current proprio across F*H steps
        for the predict call. We now match validate_on_dataset and pass no
        `state` to the predict path (state is only used by compute_kv_cache,
        which uses `self.last_action_chunk` instead). Kept as a no-op stub in
        case future logic wants a proprio-conditioned state again.
        """
        raise NotImplementedError("_build_state_tensor is no longer used")

    @torch.no_grad()
    def _warmup_kv_cache(self, raw_obs: RawObservation, cur_frame: dict) -> None:
        """Prime the DiT / streaming-VAE KV cache before the first real prediction.

        At inference chunk 0, the model's `cond` channel only has 1 valid latent
        frame (`init_latent`), while during training the cond channel always
        held the full F latent frames. This mismatch pushes chunk-0 output
        OOD, causing large pose spikes for joints far from the normalization
        center.

        Fix: feed `frame_chunk_size * 4` copies of the CURRENT observation as
        a fake "past 16 raw frames" + a static-action state (current motor pos
        held for the whole chunk). Wan VAE encodes these into 4 latent frames
        of a "static" video and writes them into the DiT KV cache. The very
        first real prediction then goes through the chunk-c>=1 code path with
        a properly populated cond history, matching the training distribution.

        Physically consistent: at session start the robot is stationary, so a
        static-video / hold-position state is a reasonable prior.

        NOTE: Wan VAE causal encoder requires its FIRST call to have T=1 (or
        T=4k+1) frames so its feat_cache is initialized correctly. We therefore
        first prime via a single-frame `_encode_obs` (which also sets
        `init_latent`), then feed 16 copies via compute_kv_cache.
        """
        need = self.frame_chunk_size * 4
        cur_pos = np.asarray(
            [float(raw_obs[k]) for k in self.motor_keys], dtype=np.float32)
        fake_state = np.broadcast_to(
            cur_pos[:, None, None],
            (self.n_motors, self.frame_chunk_size, self.action_per_frame),
        ).copy().astype(np.float32)
        try:
            # step 1: 1-frame encode to prime streaming VAE + set init_latent
            self.va.init_latent = self.va._encode_obs({"obs": [cur_frame]})
            # step 2: 16 static frames into KV cache (writes init_latent + 4 more)
            self.va.infer({
                "compute_kv_cache": True,
                "obs": [cur_frame] * need,
                "state": fake_state,
            })
            self.has_kv_cached = True
            self.last_action_chunk = fake_state
            self.warmup_done = True
            LOG.info("KV-cache warmup done (1-frame prime + 16 static frames)")
        except Exception:  # noqa: BLE001
            LOG.exception("KV-cache warmup failed (continuing without warmup)")
            # leave has_kv_cached=False so the normal chunk-0 drop-pad path
            # is still triggered as a fallback.

    @torch.no_grad()
    def _predict(self, obs: TimedObservation) -> list[TimedAction]:
        raw = obs.get_observation()
        step = obs.get_timestep()
        ts = obs.get_timestamp()

        cur_frame = self._build_frame_dict(raw)

        # ---- warmup KV cache on the very first predict of the session ----
        # Feeds 16 copies of the current frame + static-action state so the
        # DiT sees a full 4-latent-frame cond history (matching training) and
        # the model's first real prediction is NOT in the chunk-0 OOD regime.
        if not self.warmup_done:
            LOG.info("First predict of session: running KV-cache warmup ...")
            self._warmup_kv_cache(raw, cur_frame)

        # ---- visualization dump (mirror lingbot-va _encode_obs preprocessing) ----
        if self.dump_dir is not None:
            try:
                resized_unit = []  # each [3, H, W] in [0,1]
                for ck_va, img in cur_frame.items():
                    safe_ck = ck_va.replace("observation.images.", "").replace("/", "_")
                    _save_uint8_image(
                        img, self.dump_dir / f"step_{step:06d}_cam_{safe_ck}_raw.png")
                    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
                    t = F.interpolate(t.unsqueeze(0),
                                      size=(self.config.height, self.config.width),
                                      mode="bilinear", align_corners=False).squeeze(0)
                    resized_unit.append(t)
                # horizontal concat (front | wrist.left) to mirror dim-W cat in latent space
                concat_unit = torch.cat(resized_unit, dim=-1)               # [3, H, W*N]
                _save_unit_tensor(concat_unit,
                                  self.dump_dir / f"step_{step:06d}_concat.png")
                normed = concat_unit * 2.0 - 1.0
                _save_normalized_tensor(
                    normed, self.dump_dir / f"step_{step:06d}_model_input.png")
            except Exception:  # noqa: BLE001
                LOG.exception("dump preprocessing failed (continuing)")

        # If we already produced a chunk before, advance the streaming-VAE KV
        # cache with `frame_chunk_size * 4` recent sampled frames + the chunk we
        # last predicted (used as the executed action seq). Wan VAE temporal
        # stride = 4 (4 raw sampled frames -> 1 latent frame), so producing
        # `frame_chunk_size` latent frames per KV update requires
        # `frame_chunk_size * 4` sampled frames -- matches training-time chunk
        # advance and validate_on_dataset.py.
        if self.has_kv_cached and self.last_action_chunk is not None:
            need = self.frame_chunk_size * 4
            # Match training temporal sampling: stride = round(client_fps / target_fps).
            # E.g. client_fps=30, target_fps=15 -> stride=2 (keep every other obs).
            stride = max(1, int(round(self.fps / max(1, self.target_fps))))
            with self.lock:
                buf_snapshot = list(self.obs_buffer)
            # Walk from newest to oldest taking every `stride`-th frame, then reverse.
            sampled = [fd for (_, fd) in buf_snapshot[::-1][::stride]][:need][::-1]
            if len(sampled) < need:
                pad = sampled[0] if sampled else cur_frame
                sampled = ([pad] * (need - len(sampled))) + sampled
            kv_state = self.last_action_chunk
            try:
                self.va.infer({
                    "compute_kv_cache": True,
                    "obs": sampled,
                    "state": kv_state.astype(np.float32),
                })
            except Exception:
                LOG.exception("compute_kv_cache failed (continuing with stale cache)")

        # Predict next action chunk. Match validate_on_dataset: only feed
        # `obs`, do NOT pass a `state` here (state is only consumed by the
        # compute_kv_cache path during training / validation).
        t0 = time.perf_counter()
        ret = self.va.infer({
            "obs": [cur_frame],
        })
        infer_ms = (time.perf_counter() - t0) * 1000.0
        action = ret.get("action", None)
        if action is None:
            LOG.warning("VA.infer returned no action")
            return []
        # `action` shape from postprocess_action: (n_motors, F, action_per_frame)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 3 or action.shape[0] != self.n_motors:
            raise RuntimeError(
                f"unexpected action shape {action.shape}; "
                f"expected ({self.n_motors}, F, action_per_frame)")

        self.last_action_chunk = action.copy()
        first_chunk_this_session = not self.has_kv_cached
        self.has_kv_cached = True

        # Flatten (F, action_per_frame) into one time axis, then chunk to client.
        # Order: time = f * action_per_frame + j (j fast).
        a_flat = action.transpose(1, 2, 0).reshape(-1, self.n_motors)  # [F*H, C]

        if first_chunk_this_session:
            # Training pads `action_per_frame` zeros at the very start of each
            # episode's action sequence (lerobot_latent_dataset._action_post_process,
            # pad_width=frame_stride*4 which equals action_per_frame). Chunk 0's
            # first `action_per_frame` outputs correspond to that zero-padded
            # past context, NOT real predictions. Executing them on the real
            # robot causes a large jerky pose jump at session start. Drop them.
            drop = self.action_per_frame
            LOG.info("chunk 0: dropping first %d padded actions", drop)
            a_flat = a_flat[drop:]
            ts = ts + drop / float(self.fps)
            step = step + drop
        K = min(self.actions_per_chunk, a_flat.shape[0])
        dt = 1.0 / float(self.fps)
        chunk = [
            TimedAction(
                timestamp=ts + i * dt,
                timestep=step + i,
                action=torch.from_numpy(a_flat[i]),
            )
            for i in range(K)
        ]
        # state / action json dump
        if self.dump_dir is not None:
            try:
                state_vec = [float(raw[k]) for k in self.motor_keys]
                payload = {
                    "step": int(step),
                    "ts": float(ts),
                    "infer_ms": round(infer_ms, 2),
                    "state_raw": [round(v, 4) for v in state_vec],
                    "action_chunk_len": int(K),
                    "action_first": [round(float(x), 4) for x in a_flat[0]],
                    "action_mid":   [round(float(x), 4) for x in a_flat[K // 2]],
                    "action_last":  [round(float(x), 4) for x in a_flat[K - 1]],
                    "action_delta_first_minus_state":
                        [round(float(a_flat[0, j] - state_vec[j]), 4)
                         for j in range(self.n_motors)],
                }
                (self.dump_dir / f"step_{step:06d}_state_action.json").write_text(
                    json.dumps(payload, indent=2))
            except Exception:  # noqa: BLE001
                LOG.exception("dump state/action failed (continuing)")
        LOG.info(
            "step=%d infer=%.0fms chunk_len=%d a[0]=%s a[-1]=%s",
            step, infer_ms, K,
            [round(float(x), 2) for x in a_flat[0]],
            [round(float(x), 2) for x in a_flat[K - 1]],
        )
        return chunk


# ---- entrypoint --------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True,
                    help="lingbot-va config key (e.g., so_arm101 / so_arm101_genghaotian)")
    ap.add_argument("--checkpoint", default=None,
                    help="Override transformer/vae/text_encoder root dir "
                         "(must contain transformer/ vae/ tokenizer/ text_encoder/)")
    ap.add_argument("--prompt", required=True,
                    help="Task instruction string sent to T5 (one per session)")
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--port", type=int, default=15173)
    ap.add_argument("--dist-port", type=int, default=29501,
                    help="MASTER_PORT for the internal 1-rank dist group")

    ap.add_argument("--actions-per-chunk", type=int, default=16,
                    help="How many TimedActions to return per GetActions call. "
                         "Should be <= frame_chunk_size * action_per_frame.")
    ap.add_argument("--fps", type=int, default=30,
                    help="Client-side fps used to timestamp returned actions.")
    ap.add_argument("--target-fps", type=int, default=15,
                    help="Training-time sampled fps (frame_stride = fps/target_fps). "
                         "Used by the server to subsample the 30fps obs stream so "
                         "the streaming VAE sees the same temporal scale it was "
                         "trained on. Default 15 matches preprocess_so_arm101.py.")
    ap.add_argument("--cam-keys", nargs="+", default=None,
                    help="Camera keys in robot observation dict (in same order "
                         "as obs_cam_keys in lingbot-va config). Defaults to "
                         "config.obs_cam_keys (use raw cam name on the robot).")
    ap.add_argument("--motor-keys", nargs="+",
                    default=["shoulder_pan.pos", "shoulder_lift.pos",
                            "elbow_flex.pos", "wrist_flex.pos",
                            "wrist_roll.pos", "gripper.pos"])
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the startup warmup pass. Faster server boot; "
                         "first real GetActions will pay the compile/alloc cost.")
    ap.add_argument("--video-steps", type=int, default=None,
                    help="Override config.num_inference_steps (video diffusion "
                         "denoising steps). Default: from config (usually 20).")
    ap.add_argument("--action-steps", type=int, default=None,
                    help="Override config.action_num_inference_steps (action "
                         "diffusion denoising steps). Default: from config "
                         "(usually 50). Lower = faster inference.")
    ap.add_argument("--video-exec-step", type=int, default=None,
                    help="Override config.video_exec_step. -1 = run all video "
                         "denoising steps (required for correct DiT KV cache "
                         "that the action branch reads). Change only if you "
                         "know what you are doing.")
    ap.add_argument("--dump-images-dir", default=None,
                    help="If set, on every GetActions save raw per-cam PNGs, "
                         "the resized+concatenated (front|wrist) preview, the "
                         "[-1,1] normalized model_input preview, and a json "
                         "of state/predicted action under "
                         "<dir>/run_<startup_ts>/session_NNN/. Top-level "
                         "server_args.json and va_config.json snapshot the run.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    servicer = LingBotVAServicer(args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            # Allow client keepalive pings every 10s even when no RPC is
            # in-flight. Must match / be more permissive than client-side
            # keepalive_time_ms, otherwise the server will send GOAWAY and
            # the client sees "Stream removed (Socket closed)".
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 5_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10_000),
            ("grpc.http2.min_ping_interval_without_data_ms", 5_000),
            # Never close idle connections on our side.
            ("grpc.max_connection_idle_ms", 2_147_483_647),
            ("grpc.max_connection_age_ms", 2_147_483_647),
        ],
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    LOG.info("LingBot-VA async server listening on :%d", args.port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOG.info("Shutting down ...")
        servicer.shutdown_event.set()
        server.stop(grace=1.0)


if __name__ == "__main__":
    main()
