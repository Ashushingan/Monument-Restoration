"""
WEEKS 3-4: Architectural Style Classifier
==========================================
Module 1 — CNN-based classification of monument architectural style.
Uses pretrained ResNet-50 with fine-tuned classification head.

Usage:
    trainer = StyleClassifierTrainer(data_dir="data/raw", num_classes=8)
    trainer.train(epochs=30)
    style = trainer.predict("path/to/monument.jpg")
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# Import our dataset module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import get_style_loaders, STYLE_CLASSES, get_val_augmentation
from utils.metrics import compute_accuracy, plot_confusion_matrix

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────
class ArchitecturalStyleClassifier(nn.Module):
    """
    ResNet-50 backbone with custom classification head.
    Pretrained on ImageNet, fine-tuned for architectural style.
    """
    def __init__(self, num_classes=8, dropout=0.4, freeze_backbone=False):
        super().__init__()

        # Load pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Optionally freeze early layers (useful when data is scarce)
        if freeze_backbone:
            for name, param in backbone.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad = False

        # Remove final FC layer, keep feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = 2048  # ResNet-50 output dimension

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)   # [B, 2048, 1, 1]
        logits = self.classifier(features)      # [B, num_classes]
        return logits

    def get_embedding(self, x):
        """Return 2048-dim feature embedding (used by GAN for style conditioning)."""
        with torch.no_grad():
            features = self.feature_extractor(x)
            return features.squeeze(-1).squeeze(-1)  # [B, 2048]


# ─────────────────────────────────────────────
# LOSS: LABEL-SMOOTHED CROSS ENTROPY
# ─────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        # One-hot smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
class StyleClassifierTrainer:
    def __init__(self, data_dir, num_classes=8, batch_size=64,
                 lr=1e-4, save_dir="checkpoints/style",
                 image_size=512, grad_accum_steps=1):
        """
        Args:
            batch_size:        Per-step batch size. 64 is comfortable on A100 80 GB.
            image_size:        Input resolution. 512 is full resolution (original design).
            grad_accum_steps:  1 = disabled (no accumulation needed on A100).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.grad_accum_steps = grad_accum_steps
        self.image_size = image_size

        print(f"[StyleClassifier] Using device: {self.device}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            print(f"[StyleClassifier] GPU: {props.name} | VRAM: {vram_gb:.1f} GB")

        # Enable expandable memory segments to reduce fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Model — full fine-tuning, A100 has ample memory for all layers
        self.model = ArchitecturalStyleClassifier(
            num_classes=num_classes,
            freeze_backbone=False     # fine-tune entire network on A100
        ).to(self.device)

        # Data — full 512×512 resolution
        self.train_loader, self.val_loader = get_style_loaders(
            data_dir, batch_size, image_size=image_size
        )

        # Optimiser: different LR for backbone vs head
        backbone_params = list(self.model.feature_extractor.parameters())
        head_params     = list(self.model.classifier.parameters())
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},   # slower for pretrained
            {"params": head_params,     "lr": lr},
        ], weight_decay=1e-4)

        # Scheduler: cosine annealing
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-6)

        # Loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_acc = 0.0

        eff_batch = batch_size * grad_accum_steps
        print(f"[StyleClassifier] Batch={batch_size} × AccumSteps={grad_accum_steps} "
              f"→ Effective batch={eff_batch} | Image size={image_size}×{image_size}")

    def _run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(is_train):
            if is_train:
                self.optimizer.zero_grad()

            for step, (images, labels) in enumerate(
                tqdm(loader, desc="Train" if is_train else "Val", leave=False)
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Mixed precision forward pass
                with torch.autocast(device_type=self.device.type,
                                    enabled=(self.device.type == "cuda")):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if is_train:
                    # Scale loss for gradient accumulation
                    (loss / self.grad_accum_steps).backward()

                    if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(loader):
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    def train(self, epochs=30):
        print(f"\n{'='*50}")
        print(f"Training Architectural Style Classifier")
        print(f"Classes: {STYLE_CLASSES}")
        print(f"{'='*50}\n")

        for epoch in range(1, epochs + 1):
            train_loss, _ = self._run_epoch(self.train_loader, is_train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, is_train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch:02d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.2f}%")

            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": STYLE_CLASSES,
                }, self.save_dir / "best_style_classifier.pth")
                print(f"  ✓ Saved best model (acc={val_acc*100:.2f}%)")

            # Free unused GPU cache between epochs
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._plot_history()
        print(f"\nBest Val Accuracy: {self.best_acc*100:.2f}%")

    def _plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)

        ax1.plot(epochs, self.history["train_loss"], label="Train Loss")
        ax1.plot(epochs, self.history["val_loss"],   label="Val Loss")
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, [a * 100 for a in self.history["val_acc"]], color="green")
        ax2.set_title("Validation Accuracy (%)")
        ax2.set_xlabel("Epoch")

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=120)
        plt.close()
        print(f"Saved training curves → {self.save_dir}/training_curves.png")

    def load_best(self):
        ckpt = torch.load(self.save_dir / "best_style_classifier.pth", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model (epoch {ckpt['epoch']}, acc={ckpt['val_acc']*100:.2f}%)")

    def predict(self, image_path):
        """
        Predict architectural style of a single image.
        Returns: (style_name, confidence, embedding_2048d)
        """
        transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image=image)["image"].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = probs.max(dim=1)
            embedding = self.model.get_embedding(tensor)

        style = STYLE_CLASSES[class_idx.item()]
        return style, confidence.item(), embedding.cpu()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trainer = StyleClassifierTrainer(
        data_dir="data/raw",
        num_classes=len(STYLE_CLASSES),
        batch_size=64,
        lr=1e-4,
        image_size=512,
        grad_accum_steps=1,
    )
    trainer.train(epochs=30)
