# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_frank3_cfg = EasyDict(__name__='Config: VA frank3')
va_frank3_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_frank3_cfg.wan22_pretrained_model_name_or_path = "/m2v_intern2/genghaotian/models/Robbyant/lingbot-va-base"

va_frank3_cfg.attn_window = 30
va_frank3_cfg.frame_chunk_size = 4
va_frank3_cfg.env_type = 'none'

va_frank3_cfg.height = 256
va_frank3_cfg.width = 256
va_frank3_cfg.action_dim = 30
va_frank3_cfg.action_per_frame = 8
va_frank3_cfg.obs_cam_keys = [
    'observation.images.front', 'observation.images.wrist.left'
]
va_frank3_cfg.guidance_scale = 5
va_frank3_cfg.action_guidance_scale = 1

va_frank3_cfg.num_inference_steps = 20
va_frank3_cfg.video_exec_step = -1
va_frank3_cfg.action_num_inference_steps = 50

va_frank3_cfg.snr_shift = 5.0
va_frank3_cfg.action_snr_shift = 0.05

# state/action = ee_pose(xyz+quat, 7) + gripper(1) = 8 dims
va_frank3_cfg.used_action_channel_ids = list(range(0, 8))
inverse_used_action_channel_ids = [len(va_frank3_cfg.used_action_channel_ids)
                                   ] * va_frank3_cfg.action_dim
for i, j in enumerate(va_frank3_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_frank3_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_frank3_cfg.action_norm_method = 'quantiles'
# TODO: replace q01/q99 with the 8-dim values printed by script/preprocess_frank3.py
va_frank3_cfg.norm_stat = {
    "q01": [
        0.0,  # ee_x
        0.0,  # ee_y
        0.0,  # ee_z
        -1.0,  # ee_qx
        -1.0,  # ee_qy
        -1.0,  # ee_qz
        -1.0,  # ee_qw
        0.0,  # gripper.pos
    ] + [0.] * 22,
    "q99": [
        1.0,  # ee_x
        1.0,  # ee_y
        1.0,  # ee_z
        1.0,  # ee_qx
        1.0,  # ee_qy
        1.0,  # ee_qz
        1.0,  # ee_qw
        1.0,  # gripper.pos
    ] + [0.] * 22,
}
