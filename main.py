import torchvision
import torch
import os
import random
import numpy as np
import pandas as pd
import datetime
from configs.watercolor2k_config import WaterColor2kConfig
from datasets.datasets import VOCStyleDataset
from model import build_faster_rcnn
from datasets.dataloader import get_dataloader
from utils.utils import set_seed
from engine.train_loop import Trainer
from engine.evaluator import Evaluator
from utils.optimizer import get_optimizer
from utils.visualizer import visualize_prediction, visualize_ground_truth_and_predictions
from datasets.transforms import get_transform


def main():
    config = WaterColor2kConfig(mode="finetune") # Mode options :| "zeroshot" | "finetune" | "scratch" |
    set_seed(config.seed)

    # Initialize model
    model = build_faster_rcnn(config).to(config.device)
    # Load dataloader
    train_loader = get_dataloader(config, "train")
    test_loader = get_dataloader(config, "test")

    ann_path = f"{config.data_root}/annotations/{config.dataset}_test.json"
    img_dir = f"{config.data_root}/JPEGImages"

    dataset = VOCStyleDataset(
        img_folder=img_dir,
        ann_file=ann_path,
        transforms=get_transform(train=False)
    )

    # Initialize Training Loop
    if config.eval_only:
        print(f"[{config.mode.upper()}] Evaluation only (no training).")
        evaluator = Evaluator(model, test_loader, config.device)
        evaluator.evaluate()
    else:
        print(f"[{config.mode.upper()}] Training mode.")
        train_loader = get_dataloader(config, split = "train")
        optimizer = get_optimizer(config, model)
        trainer = Trainer(model, optimizer, train_loader, test_loader, config.device)
        trainer.train(epochs=config.epoch, save_path=config.output_dir)

    if config.mode != 'zeroshot':
        models = []
        model.load_state_dict(torch.load(f"{config.output_dir}/best_model.pth"))
        models.append(model)
    else:
        models = []
        models.append(model)
    visualize_ground_truth_and_predictions(models, "zeroshot", dataset, config.device, config, target_class="person", start_idx=60, score_threshold=0.5, save_path=config.save_path)
        
if __name__ == '__main__':
    main()