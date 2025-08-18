#!/usr/bin/env python3

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
csts_path = os.path.expanduser("~/junseok/ego-dataset-train/CSTS")
sys.path.append(str(csts_path))

import torch
import yaml
from src.data_loader import Ego4dDataset

from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args, load_config


from slowfast.models.build import build_model
from slowfast.models.optimizer import construct_optimizer
from slowfast.utils.checkpoint import load_train_checkpoint, save_checkpoint
from slowfast.utils.meters import TrainGazeMeter, ValGazeMeter, TestGazeMeter
from slowfast.utils import distributed as du

from tools.train_avgaze_net import train
from tools.test_avgaze_net import test


def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()
