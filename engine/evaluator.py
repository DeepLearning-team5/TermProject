import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataloader, device, logger=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger

        self.mAP_50 = MeanAveragePrecision(iou_thresholds=[0.5])
        self.mAP_75 = MeanAveragePrecision(iou_thresholds=[0.75])
        self.mAP_90 = MeanAveragePrecision(iou_thresholds=[0.9])

    def evaluate(self, epoch=None):
        self.model.eval()
        self.mAP_50.reset()
        self.mAP_75.reset()
        self.mAP_90.reset()

        pbar = tqdm(self.dataloader, desc="Evaluating")
        for images, targets in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]


            self.mAP_50.update(outputs, targets)
            self.mAP_75.update(outputs, targets)
            self.mAP_90.update(outputs, targets) 
        
        mAP_50 = mAP_50.compute()['map'].item()
        mAP_75 = mAP_75.compute()['map'].item()
        mAP_90 = mAP_90.compute()['map'].item()

        print("========== Evaluation Result ==========")
        print(f"    mAP@0.50: {mAP_50['map']:.4f}")
        print(f"    mAP@0.75: {mAP_75['map']:.4f}")
        print(f"    mAP@0.90: {mAP_90['map']:.4f}")

        if self.logger:
            self.logger.log({
                "val/mAP@0.50": mAP_50,
                "val/mAP@0.75": mAP_75,
                "val/mAP@0.90": mAP_90,
                "epoch": epoch
            })
        
        return mAP_50
