from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import os
import torch

class WatercolorCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(WatercolorCocoDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(WatercolorCocoDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x,y,x+w,y+h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        if self._transforms is not None:
            img = self._transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)