
import torch
from tqdm import tqdm
from engine.evaluator import Evaluator

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader, device, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = logger

        self.evaluator = Evaluator(self.model, self.test_loader, self.device, self.logger)

    def train(self, epochs, eval_fn = None, save_path=None):
        best_score = -1.0
        early_stopping_counter = 0

        for epoch in range(epochs):
            loss = self.train_one_epoch(epoch)

            if self.lr_schedular:
                self.lr_schedular.step()
            
            if self.evaluator:
                
                score = self.evaluator.evaluate(epoch)

                if score > best_score:
                    best_score = score
                    early_stopping_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), f"{save_path}/best_model.pth")
                        print(f"[Checkpoint] Best model saved (score: {score:.4f})")
                else:
                    early_stopping_counter += 1


                if early_stopping_counter >= self.early_stopping:
                    print(f"[Early Stop] No Improvement for {self.early_stopping} epochs.")
                    break
            
            if save_path:
                torch.save(self.model.state_dict(), f"{save_path}/model_epoch_{epoch}.pth")
            
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}")
        for step, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss = loss.item())

            if self.logger:
                self.logger.log({
                    "train/loss_step": loss.item(),
                    "train/loss_classifier": loss_dict["loss_classifier"].item(),
                    "train/loss_box_reg": loss_dict["loss_box_reg"].item(),
                    "train/loss_objectness": loss_dict["loss_objectness"].item(),
                    "train/loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"].item(),
                    "step": step + epoch * len(self.train_loader)
                })

            avg_loss = total_loss / len(self.train_loader)
            print(f"[Train] Avg Loss: {avg_loss:.4f}")

            if self.logger:
                self.logger.log({
                    "train/loss_epoch": avg_loss,
                    "epoch": epoch
                })

            return avg_loss
