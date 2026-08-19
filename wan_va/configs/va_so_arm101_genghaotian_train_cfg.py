# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_so_arm101_genghaotian_cfg import va_so_arm101_genghaotian_cfg
import os

va_so_arm101_genghaotian_train_cfg = EasyDict(__name__='Config: VA so_arm101 genghaotian train')
va_so_arm101_genghaotian_train_cfg.update(va_so_arm101_genghaotian_cfg)

va_so_arm101_genghaotian_train_cfg.dataset_path = '/m2v_intern/genghaotian/lingbot-va/data/lerobot_so_arm101_genghaotian_v21'
va_so_arm101_genghaotian_train_cfg.empty_emb_path = os.path.join(va_so_arm101_genghaotian_train_cfg.dataset_path, 'empty_emb.pt')
va_so_arm101_genghaotian_train_cfg.enable_wandb = True
va_so_arm101_genghaotian_train_cfg.load_worker = 8
va_so_arm101_genghaotian_train_cfg.save_interval = 500
va_so_arm101_genghaotian_train_cfg.gc_interval = 50
va_so_arm101_genghaotian_train_cfg.cfg_prob = 0.1

# Training parameters
va_so_arm101_genghaotian_train_cfg.learning_rate = 1e-5
va_so_arm101_genghaotian_train_cfg.beta1 = 0.9
va_so_arm101_genghaotian_train_cfg.beta2 = 0.95
va_so_arm101_genghaotian_train_cfg.weight_decay = 1e-1
va_so_arm101_genghaotian_train_cfg.warmup_steps = 50
va_so_arm101_genghaotian_train_cfg.batch_size = 1
va_so_arm101_genghaotian_train_cfg.gradient_accumulation_steps = 10
va_so_arm101_genghaotian_train_cfg.num_steps = 4000
