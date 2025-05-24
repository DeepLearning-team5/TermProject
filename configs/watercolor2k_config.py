import torch

class WaterColor2kConfig:
    def __init__(self, mode="zeroshot"):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset Settings
        self.dataset = 'watercolor'
        self.num_classes = 8
        self.image_size = (320, 320)
        self.num_workers = 4

        # Training Settings
        self.batch_size = 4
        self.epoch = 10
        self.lr = 0.0001
        self.weight_decay=1e-4
        self.optimizer='Adam'


        # Model Settings
        self.model = 'fasterrcnn_resnet50_fpn'
        self.pretrained = True
        self.backbone_freeze = True
        self.head_random_init = False

        # Eval / Mode
        self.eval_only = False
        self.mode = mode  # "zeroshot" | "finetune" | "scratch"
        self.configure_mode()

        # Paths
        self.data_root = './data/watercolor'
        self.output_dir = f'./checkpoints/watercolor/{mode}'
        self.save_path = f'./visualization/watercolor/{mode}'

        # Dataset Information
        self.CLASSES = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']

    def configure_mode(self):
        if self.mode == "zeroshot":
            self.eval_only = True
            self.backbone_freeze = True
            self.head_random_init = False

        elif self.mode == "finetune":
            self.eval_only = False
            self.backbone_freeze = True
            self.head_random_init = False

        elif self.mode == "scratch":
            self.eval_only = False
            self.backbone_freeze = True
            self.head_random_init = True

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
