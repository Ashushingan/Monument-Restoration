"""
WEEKS 5-6: U-Net Damage Segmentation
======================================
Module 2 — Detects and segments damaged/missing regions in monument images.
Produces binary damage masks (1=damaged, 0=intact).

Architecture: U-Net with ResNet-34 encoder (pretrained on ImageNet)
Loss: Combo of Binary Cross Entropy + Dice Loss for imbalanced masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import get_inpainting_loaders


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice loss for binary segmentation — handles class imbalance well."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedSegLoss(nn.Module):
    """BCE + Dice — stable training for small damaged regions."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + self.dice_w * self.dice(pred, target)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class DamageSegmentationModel(nn.Module):
    """
    U-Net with ResNet-34 encoder.
    Input:  [B, 3, 512, 512]  — RGB monument image
    Output: [B, 1, 512, 512]  — damage probability map (logits)
    """
    def __init__(self, encoder_name="resnet34", pretrained=True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None,   # raw logits; sigmoid applied in loss/inference
        )

    def forward(self, x):
        return self.unet(x)

    def predict_mask(self, x, threshold=0.5):
        """Return binary mask from image tensor."""
        logits = self.forward(x)
        probs  = torch.sigmoid(logits)
        return (probs > threshold).float()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def iou_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return (intersection / (union + 1e-6)).mean().item()

def dice_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return (2 * intersection / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6)).mean().item()


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
class DamageSegmentationTrainer:
    def __init__(self, data_dir, batch_size=16, lr=1e-3,
                 save_dir="checkpoints/segmentation"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DamageSegmentation] Using device: {self.device}")

        self.model = DamageSegmentationModel().to(self.device)
        self.train_loader, self.val_loader = get_inpainting_loaders(data_dir, batch_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=5,
                                           factor=0.5)
        self.criterion = CombinedSegLoss(bce_weight=0.4, dice_weight=0.6)

        self.history = {"train_loss": [], "val_iou": [], "val_dice": []}
        self.best_iou = 0.0

    def _run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, total_iou, total_dice, n_batches = 0.0, 0.0, 0.0, 0

        with torch.set_grad_enabled(is_train):
            for batch in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
                images = batch["damaged"].to(self.device)
                masks  = batch["mask"].to(self.device)     # [B, 1, H, W]

                pred_logits = self.model(images)
                loss = self.criterion(pred_logits, masks)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_iou  += iou_score(pred_logits, masks)
                total_dice += dice_score(pred_logits, masks)
                n_batches  += 1

        return total_loss / n_batches, total_iou / n_batches, total_dice / n_batches

    def train(self, epochs=40):
        print(f"\n{'='*50}")
        print("Training U-Net Damage Segmentation Model")
        print(f"{'='*50}\n")

        for epoch in range(1, epochs + 1):
            train_loss, _, _ = self._run_epoch(self.train_loader, is_train=True)
            _, val_iou, val_dice = self._run_epoch(self.val_loader, is_train=False)
            self.scheduler.step(val_iou)

            self.history["train_loss"].append(train_loss)
            self.history["val_iou"].append(val_iou)
            self.history["val_dice"].append(val_dice)

            print(f"Epoch [{epoch:02d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | "
                  f"Val Dice: {val_dice:.4f}")

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
                }, self.save_dir / "best_damage_segmenter.pth")
                print(f"  ✓ Saved best model (IoU={val_iou:.4f})")

        self._plot_history()
        self._save_sample_predictions()

    def _plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        ax1.plot(epochs, self.history["train_loss"], label="Train Loss")
        ax1.set_title("Segmentation Training Loss")
        ax1.set_xlabel("Epoch")

        ax2.plot(epochs, self.history["val_iou"],  label="IoU",  color="blue")
        ax2.plot(epochs, self.history["val_dice"], label="Dice", color="orange")
        ax2.set_title("Validation Metrics")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / "seg_training_curves.png", dpi=120)
        plt.close()

    def _save_sample_predictions(self, n=4):
        """Visualise model predictions on validation set."""
        self.model.eval()
        batch = next(iter(self.val_loader))
        images = batch["damaged"][:n].to(self.device)
        gt_masks = batch["mask"][:n]

        with torch.no_grad():
            pred_masks = self.model.predict_mask(images).cpu()

        # Denorm images for display (reverse the [-1,1] normalisation)
        def denorm(t):
            return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        rows = []
        for i in range(n):
            img_show  = denorm(images[i].cpu())
            gt_show   = (gt_masks[i, 0].numpy() * 255).astype(np.uint8)
            pred_show = (pred_masks[i, 0].numpy() * 255).astype(np.uint8)
            gt_rgb    = cv2.cvtColor(gt_show,   cv2.COLOR_GRAY2RGB)
            pred_rgb  = cv2.cvtColor(pred_show, cv2.COLOR_GRAY2RGB)
            row = np.hstack([img_show, gt_rgb, pred_rgb])
            rows.append(row)

        grid = np.vstack(rows)
        out_path = self.save_dir / "sample_predictions.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Saved sample predictions → {out_path}")

    def load_best(self):
        ckpt = torch.load(self.save_dir / "best_damage_segmenter.pth",
                          map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best segmenter (epoch {ckpt['epoch']}, IoU={ckpt['val_iou']:.4f})")
        return self.model

    def segment(self, image_bgr: np.ndarray, threshold=0.5):
        """
        Run inference on a single OpenCV BGR image.
        Returns: binary mask numpy array [H, W], uint8
        """
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            mask = self.model.predict_mask(tensor, threshold=threshold)

        mask_np = mask[0, 0].cpu().numpy()  # [512, 512] float
        # Resize back to original image dimensions
        orig_h, orig_w = image_bgr.shape[:2]
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return (mask_resized * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trainer = DamageSegmentationTrainer(
        data_dir="data/raw",
        batch_size=8,
        lr=1e-3,
    )
    trainer.train(epochs=40)
