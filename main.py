import torchvision
import torch
import os
import random
import numpy as np
import pandas as pd
import datetime
from configs.watercolor2k_config import WaterColor2kConfig
from model import build_faster_rcnn
from datasets.dataloader import get_dataloader
from utils.utils import set_seed
from engine.train_loop import Trainer
from engine.evaluator import Evaluator
from utils.optimizer import get_optimizer


def main():
    config = WaterColor2kConfig(mode="zeroshot") # Mode options :| "zeroshot" | "finetune" | "scratch" |
    set_seed(config.seed)

    # Initialize model
    model = build_faster_rcnn(config).to(config.device)
    # Load dataloader
    train_loader = get_dataloader(config, "train")
    test_loader = get_dataloader(config, "test")


    # Initialize Training Loop
    if config.eval_only:
        print(f"[{config.mode.upper()}] Evaluation only (no training).")
        evaluator = Evaluator(model, test_loader, config.device)
        evaluator.evaluate()
    else:
        print(f"[{config.mode.upper()}] Training mode.")
        train_loader = get_dataloader(config, mode = "train")
        optimizer = get_optimizer(config, model)
        trainer = Trainer(model, optimizer, train_loader, test_loader, config.device)
        trainer.train(epochs=config.epoch, save_path=config.output_dir)

        
if __name__ == '__main__':
    main()