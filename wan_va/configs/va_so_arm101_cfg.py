# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_so_arm101_cfg = EasyDict(__name__='Config: VA so_arm101')
va_so_arm101_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_so_arm101_cfg.wan22_pretrained_model_name_or_path = "/m2v_intern2/genghaotian/models/Robbyant/lingbot-va-base"

va_so_arm101_cfg.attn_window = 30
va_so_arm101_cfg.frame_chunk_size = 4
va_so_arm101_cfg.env_type = 'none'

va_so_arm101_cfg.height = 256
va_so_arm101_cfg.width = 256
va_so_arm101_cfg.action_dim = 30
va_so_arm101_cfg.action_per_frame = 8
va_so_arm101_cfg.obs_cam_keys = [
    'observation.images.front', 'observation.images.wrist.left'
]
va_so_arm101_cfg.guidance_scale = 5
va_so_arm101_cfg.action_guidance_scale = 1

va_so_arm101_cfg.num_inference_steps = 20
va_so_arm101_cfg.video_exec_step = -1
va_so_arm101_cfg.action_num_inference_steps = 50

va_so_arm101_cfg.snr_shift = 5.0
va_so_arm101_cfg.action_snr_shift = 0.05

va_so_arm101_cfg.used_action_channel_ids = list(range(0, 6))
inverse_used_action_channel_ids = [len(va_so_arm101_cfg.used_action_channel_ids)
                                   ] * va_so_arm101_cfg.action_dim
for i, j in enumerate(va_so_arm101_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_so_arm101_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_so_arm101_cfg.action_norm_method = 'quantiles'
va_so_arm101_cfg.norm_stat = {
    "q01": [
        -48.571430,
        -104.483513,
        -71.047033,
        48.351650,
        -96.659340,
        0.831255,
    ] + [0.] * 24,
    "q99": [
        49.362637,
        40.835163,
        96.615387,
        102.065933,
        -35.912086,
        16.126350,
    ] + [0.] * 24,
}
