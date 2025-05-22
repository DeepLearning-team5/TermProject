import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(config):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if config.pretrained else None
    model = fasterrcnn_resnet50_fpn(min_size=64, max_size=224, weights=weights)

    if config.backbone_freeze:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # 3. Replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)

    # 4. (Optional) Random initialize head — for semi-scratch
    if config.head_random_init:
        torch.nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
        torch.nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, 0)
        torch.nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.001)
        torch.nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, 0)

    return model