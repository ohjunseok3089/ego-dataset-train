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
from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args, load_config

# ====== 사용자 정의 데이터셋 클래스 (동일할 경우 CSTS logic 복사) ======
from src.data_loader import Ego4dDataset

# ====== CSTS 모델, 옵티마이저, 메터, 기타 유틸 ======
from slowfast.models.build import build_model
from slowfast.models.optimizer import construct_optimizer
from slowfast.utils.checkpoint import load_train_checkpoint, save_checkpoint
from slowfast.utils.meters import TrainGazeMeter, ValGazeMeter, TestGazeMeter
from slowfast.utils import distributed as du

def train(cfg):
    du.init_distributed_training(cfg)
    train_dataset = Ego4dDataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=True,
    )
    model = build_model(cfg)
    optimizer = construct_optimizer(model, cfg)
    start_epoch = load_train_checkpoint(cfg, model, optimizer)
    train_meter = TrainGazeMeter(len(train_loader), cfg)
    model.train()

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        train_meter.reset()
        for cur_iter, (inputs, labels, index, extra_data) in enumerate(train_loader):
            inputs = [inp.cuda(non_blocking=True) for inp in inputs]
            labels = labels.cuda()
            preds = model(inputs)
            loss = torch.nn.functional.kl_div(torch.log_softmax(preds, dim=-1), labels, reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update_stats(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                mb_size=inputs.size(0)
            )
            if (cur_iter + 1) % cfg.LOG_PERIOD == 0:
                train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.log_epoch_stats(cur_epoch)
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
    print("TRAINING DONE.")

def test(cfg):
    du.init_distributed_training(cfg)
    test_dataset = Ego4dDataset(cfg, mode="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False,
    )
    model = build_model(cfg)
    _ = load_train_checkpoint(cfg, model)
    model.eval()
    test_meter = TestGazeMeter(len(test_loader), cfg)
    with torch.no_grad():
        for cur_iter, (inputs, labels, index, extra_data) in enumerate(test_loader):
            inputs = [inp.cuda(non_blocking=True) for inp in inputs]
            labels = labels.cuda()
            preds = model(inputs)
            # metrics update etc
            test_meter.update_stats(preds, labels)
        test_meter.finalize_metrics()
    print("TESTING DONE.")

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
