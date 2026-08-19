#!/usr/bin/env bash
# Launch LingBot-VA real-robot server for SO-ARM101.
#
# Edit CONFIG_NAME / CHECKPOINT / PROMPT below, then run:
#   bash evaluation/real_robot/launch_so_arm101_server.sh
#
# After this script is running, on the same host start the WS tunnel:
#   python /m2v_intern/tujiahang/Projects/lerobot/tools/ws_tcp_tunnel.py server \
#       --listen-port 15174 --target-port 15173
# and on the robot host run the standard lerobot async client (see
# so_arm101_server.py header for the exact command).

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-so_arm101_genghaotian}"
CHECKPOINT="${CHECKPOINT:-}"        # optional override; leave empty to use config's path
PROMPT="${PROMPT:-Place the black bottle cap into the white paper cup}"
PORT="${PORT:-15173}"
DIST_PORT="${DIST_PORT:-29501}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
FPS="${FPS:-30}"
SAVE_ROOT="${SAVE_ROOT:-visualization/real_robot}"

# Camera keys here match the raw observation dict published by the lerobot
# robot_client. Edit if your robot publishes different names (the order must
# stay aligned with config.obs_cam_keys: front first, wrist.left second).
CAM_KEYS=(
    "observation.images.front"
    "observation.images.wrist.left"
)

ARGS=(
    --config-name "$CONFIG_NAME"
    --prompt "$PROMPT"
    --port "$PORT"
    --dist-port "$DIST_PORT"
    --actions-per-chunk "$ACTIONS_PER_CHUNK"
    --fps "$FPS"
    --save-root "$SAVE_ROOT"
    --cam-keys "${CAM_KEYS[@]}"
)
if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
fi

cd "$(dirname "$0")/../.."
exec python evaluation/real_robot/so_arm101_server.py "${ARGS[@]}"
