#!/usr/bin/env python
"""Synchronous SO-ARM101 client for the LingBot-VA async inference server.

This is a minimal RobotWin-style replacement for
`python -m lerobot.async_inference.robot_client`. It talks the same gRPC
protocol as `lerobot.transport.services_pb2` so it plugs into our
`so_arm101_server.py` unchanged.

Key differences vs the official lerobot async client:
  * Single thread, strict obs->action = 1:1 (mirrors RobotWin eval flow).
  * No `action_queue` / no back-pressure / no `chunk_size_threshold`.
    Each iteration: send 1 obs -> poll GetActions until server returns a
    chunk -> execute EACH action, capturing + sending 1 fresh obs after
    every executed action. This guarantees the server always receives a
    contiguous stream of frames aligned with what training saw.
  * No async streaming of obs; we do NOT keep asking for actions while the
    previous chunk executes.

Dependencies (all already installed on the robot host that runs the old
launcher):
  * `lerobot` (for SO101Follower + OpenCVCamera; robot bring-up unchanged)
  * `grpcio` + `lerobot.transport.services_pb2*` (already present)
  * torch, numpy, PIL, pickle (all stdlib / existing)
NO new packages introduced.

Example (mirrors the old launcher):
  python evaluation/real_robot/so_arm101_client.py \\
      --robot-port /dev/ttyACM1 \\
      --robot-id tjh_follower_arm \\
      --cam-front-index 0 --cam-wrist-index 2 \\
      --width 1280 --height 720 \\
      --server 127.0.0.1:8080 \\
      --prompt "Place the black bottle cap into the white paper cup" \\
      --fps 30
"""
from __future__ import annotations

import argparse
import logging
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import torch

# -----------------------------------------------------------------------------
# 1) lerobot imports (all "cheap" -- pure Python; no additional pip deps).
# -----------------------------------------------------------------------------
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks

# TransferState constants
_TS_BEGIN = services_pb2.TransferState.TRANSFER_BEGIN
_TS_MIDDLE = services_pb2.TransferState.TRANSFER_MIDDLE
_TS_END = services_pb2.TransferState.TRANSFER_END

LOG = logging.getLogger("so_arm101_client")


# -----------------------------------------------------------------------------
# 2) TimedObservation / TimedAction.
# The lerobot install on the robot host already ships these under
# `lerobot.async_inference.helpers`. Use them directly so pickle can round-trip
# to the server (whose stub is registered at the same module path). If for some
# reason the module is absent, fall back to a local stub with the same class
# path.
# -----------------------------------------------------------------------------
try:
    from lerobot.async_inference.helpers import (  # type: ignore
        TimedObservation,
        TimedAction,
    )
    _USING_LEROBOT_HELPERS = True
except Exception:
    _USING_LEROBOT_HELPERS = False

    @dataclass
    class TimedData:
        timestamp: float
        timestep: int
        def get_timestamp(self): return self.timestamp
        def get_timestep(self): return self.timestep

    @dataclass
    class TimedAction(TimedData):
        action: torch.Tensor
        def get_action(self): return self.action

    @dataclass
    class TimedObservation(TimedData):
        observation: dict[str, Any]
        must_go: bool = False
        def get_observation(self): return self.observation

    for _cls in (TimedData, TimedAction, TimedObservation):
        _cls.__module__ = "lerobot.async_inference.helpers"


# -----------------------------------------------------------------------------
# 3) Robot bring-up (SO101 + 2x OpenCV cameras).
# -----------------------------------------------------------------------------
def make_robot(
    port: str, robot_id: str,
    cam_front_index: int, cam_wrist_index: int,
    width: int, height: int, fps: int,
    fourcc: str | None = None,
) -> SO101Follower:
    cam_kwargs: dict = dict(width=width, height=height, fps=fps)
    if fourcc:
        cam_kwargs["fourcc"] = fourcc
    cam_cfg = {
        "observation.images.front": OpenCVCameraConfig(
            index_or_path=cam_front_index, **cam_kwargs),
        "observation.images.wrist.left": OpenCVCameraConfig(
            index_or_path=cam_wrist_index, **cam_kwargs),
    }
    cfg = SO101FollowerConfig(port=port, id=robot_id, cameras=cam_cfg,
                              use_degrees=True)
    robot = SO101Follower(cfg)
    LOG.info("Connecting to robot on %s (id=%s) ...", port, robot_id)
    robot.connect()
    LOG.info("Robot connected. cameras=%s", list(cam_cfg.keys()))
    return robot


# -----------------------------------------------------------------------------
# 4) gRPC helpers -- pickle a Python object, chunk it into TRANSFER_* messages.
# -----------------------------------------------------------------------------
def pickle_and_send(stub_send_fn, obj: Any) -> None:
    buf = pickle.dumps(obj)
    iterator = send_bytes_in_chunks(buf, services_pb2.Observation,
                                    log_prefix="[obs]", silent=True)
    stub_send_fn(iterator)


# -----------------------------------------------------------------------------
# 5) Main synchronous control loop.
# -----------------------------------------------------------------------------
class So101Client:
    MOTOR_KEYS = [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]
    CAM_KEYS = ["observation.images.front", "observation.images.wrist.left"]

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = make_robot(
            args.robot_port, args.robot_id,
            args.cam_front_index, args.cam_wrist_index,
            args.width, args.height, args.fps,
            fourcc=args.fourcc,
        )
        opts = [
            ("grpc.max_send_message_length", 100 << 20),
            ("grpc.max_receive_message_length", 100 << 20),
        ]
        self.channel = grpc.insecure_channel(args.server, options=opts)
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        LOG.info("gRPC channel to %s", args.server)

        self.step = 0
        self._stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *a) -> None:
        LOG.warning("SIGINT received, stopping after current action ...")
        self._stop = True

    # ---- session setup -----------------------------------------------------
    def wait_for_ready(self, timeout_s: float = 30.0) -> None:
        deadline = time.time() + timeout_s
        while True:
            try:
                self.stub.Ready(services_pb2.Empty(), timeout=2.0)
                LOG.info("Server ready.")
                return
            except grpc.RpcError as e:
                if time.time() > deadline:
                    raise RuntimeError(f"Server not ready: {e}") from e
                LOG.info("Waiting for server ...")
                time.sleep(1.0)

    def send_policy_setup(self) -> None:
        """Server unpickles this permissively; sending a minimal dict is fine.
        The server keys off the pickle bytes to (a) reset the VA session and
        (b) dump `client_policy_setup.{pkl,json}` for debugging.
        """
        payload = {
            "policy_type":              "lingbot_va",
            "pretrained_name_or_path":  "(client-managed)",
            "actions_per_chunk":        None,     # server decides
            "device":                   "cuda",
            "rename_map":               {},
            "lerobot_features":         {},
            "prompt":                   self.args.prompt,
        }
        setup = services_pb2.PolicySetup(data=pickle.dumps(payload))
        LOG.info("SendPolicyInstructions (new session, prompt=%r)", self.args.prompt)
        self.stub.SendPolicyInstructions(setup)

    # ---- observation / action --------------------------------------------
    def capture_obs(self, must_go: bool = False) -> TimedObservation:
        raw = self.robot.get_observation()
        # SO101Follower returns {"<motor>.pos": float, <cam_key>: HxWx3 uint8}.
        # Attach task string (some server implementations look at it).
        raw["task"] = self.args.prompt
        return TimedObservation(
            timestamp=time.time(),
            timestep=self.step,
            observation=raw,
            must_go=must_go,
        )

    def send_obs(self, obs: TimedObservation) -> None:
        pickle_and_send(self.stub.SendObservations, obs)

    def wait_for_chunk(self, poll_s: float = 0.05,
                       rpc_timeout_s: float = 120.0,
                       total_timeout_s: float = 300.0) -> list[TimedAction]:
        """Block until server returns a non-empty action chunk.

        The first predict of a session includes KV-cache warmup + a full
        diffusion pass which can take 20-40s on a single GPU, so
        `rpc_timeout_s` defaults to 120s. Later chunks are ~1-2s each.
        """
        deadline = time.time() + total_timeout_s
        while not self._stop:
            try:
                ret = self.stub.GetActions(services_pb2.Empty(),
                                           timeout=rpc_timeout_s)
            except grpc.RpcError as e:
                LOG.warning("GetActions error: %s (retrying)", e)
                time.sleep(poll_s)
                continue
            if ret.data:
                return pickle.loads(ret.data)
            if time.time() > deadline:
                raise TimeoutError("Server never returned actions.")
            time.sleep(poll_s)
        return []

    def apply_action(self, action_vec: np.ndarray) -> None:
        assert action_vec.shape == (len(self.MOTOR_KEYS),), \
            f"expected {len(self.MOTOR_KEYS)} motors, got {action_vec.shape}"
        action_dict = {k: float(v) for k, v in zip(self.MOTOR_KEYS, action_vec)}
        self.robot.send_action(action_dict)

    # ---- main loop --------------------------------------------------------
    def run(self) -> None:
        self.wait_for_ready()
        self.send_policy_setup()

        dt = 1.0 / float(self.args.fps)
        LOG.info("Starting sync control loop (dt=%.4fs, fps=%d)",
                 dt, self.args.fps)

        # Warm up cameras / bus by taking one initial obs and pushing it once.
        # Server needs at least one obs before GetActions returns anything.
        initial = self.capture_obs(must_go=True)
        self.send_obs(initial)

        chunk_idx = 0
        while not self._stop:
            if self.args.max_steps and self.step >= self.args.max_steps:
                LOG.info("Reached --max-steps=%d, stopping.", self.args.max_steps)
                break

            LOG.info("=== chunk %d @ step %d ===", chunk_idx, self.step)
            chunk = self.wait_for_chunk()
            if not chunk:
                break
            LOG.info("Received %d actions (timestep %d..%d)",
                     len(chunk), chunk[0].get_timestep(), chunk[-1].get_timestep())

            for i, ta in enumerate(chunk):
                if self._stop:
                    break
                t0 = time.perf_counter()
                vec = ta.get_action().detach().cpu().numpy().astype(np.float32)
                self.apply_action(vec)

                # After executing, capture a FRESH obs and push it.
                # This is the RobotWin-style strict obs<->action alignment.
                # `must_go=True` on the last action ensures the server does
                # not skip processing before the next GetActions call.
                is_last = (i == len(chunk) - 1)
                obs = self.capture_obs(must_go=is_last)
                self.send_obs(obs)
                self.step += 1

                # Pace at ~fps
                elapsed = time.perf_counter() - t0
                sleep_s = dt - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -0.02:
                    LOG.warning("loop lagging: step %d took %.3fs (target %.3fs)",
                                self.step, elapsed, dt)
            chunk_idx += 1

        LOG.info("Client stopping. Total steps=%d chunks=%d", self.step, chunk_idx)

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass
        try:
            self.robot.disconnect()
        except Exception:
            LOG.exception("robot disconnect failed")


# -----------------------------------------------------------------------------
# 6) CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-port", default="/dev/ttyACM1")
    ap.add_argument("--robot-id", default="tjh_follower_arm")
    ap.add_argument("--cam-front-index", type=int, default=0)
    ap.add_argument("--cam-wrist-index", type=int, default=2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default=None,
                    help="4-char FOURCC code, e.g. MJPG / YUYV. Default None = auto-detect.")
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Stop after N action steps (0 = infinite / Ctrl+C).")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    client = So101Client(args)
    try:
        client.run()
    finally:
        client.close()


if __name__ == "__main__":
    main()
