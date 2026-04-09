"""
WEEKS 1-2: Dataset Collection and Synthetic Damage Generation
=============================================================
This module handles:
- Dataset structure and loading
- Synthetic damage mask generation (random polygons, brush strokes, erosion)
- Image augmentation pipeline
- Dataset splitting
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import json


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE = 512
DAMAGE_RATIO_MIN = 0.05   # min 5% of image damaged
DAMAGE_RATIO_MAX = 0.40   # max 40% of image damaged

# Architectural style labels (customise as needed)
STYLE_CLASSES = [
    "Dravidian",
    "Nagara",
    "Vesara",
    "Islamic",
    "Buddhist",
    "Colonial",
    "Rock_Cut",
    "Stepwell"
]


# ─────────────────────────────────────────────
# SYNTHETIC DAMAGE MASK GENERATOR
# ─────────────────────────────────────────────
class DamageMaskGenerator:
    """
    Generates realistic synthetic damage masks for monument images.
    Combines multiple damage types: cracks, missing chunks, erosion patches.
    """

    def __init__(self, image_size=IMAGE_SIZE):
        self.size = image_size

    def _random_polygon_mask(self, n_polygons=3):
        """Simulate missing/broken-off stone chunks."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        for _ in range(n_polygons):
            n_pts = random.randint(4, 10)
            pts = np.array([
                [random.randint(0, self.size), random.randint(0, self.size)]
                for _ in range(n_pts)
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _random_brush_stroke_mask(self, n_strokes=5):
        """Simulate cracks and surface erosion lines."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        for _ in range(n_strokes):
            x1, y1 = random.randint(0, self.size), random.randint(0, self.size)
            x2, y2 = random.randint(0, self.size), random.randint(0, self.size)
            thickness = random.randint(5, 40)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            # Add slight curvature using intermediate control point
            cx, cy = random.randint(0, self.size), random.randint(0, self.size)
            pts = np.array([[x1, y1], [cx, cy], [x2, y2]], dtype=np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness)
        return mask

    def _random_ellipse_mask(self, n_ellipses=4):
        """Simulate circular weathering / water damage."""
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        for _ in range(n_ellipses):
            cx, cy = random.randint(0, self.size), random.randint(0, self.size)
            ax = random.randint(20, 120)
            ay = random.randint(20, 120)
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
        return mask

    def _random_noise_mask(self):
        """Simulate fine surface grain damage / spalling."""
        noise = np.random.rand(self.size, self.size)
        threshold = random.uniform(0.7, 0.92)
        mask = (noise > threshold).astype(np.uint8) * 255
        # Dilate to create connected regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def generate(self, damage_type="mixed"):
        """
        Generate a binary damage mask.
        damage_type: "polygon" | "brush" | "ellipse" | "noise" | "mixed"
        Returns: uint8 numpy array, 255=damaged, 0=intact
        """
        if damage_type == "polygon":
            mask = self._random_polygon_mask()
        elif damage_type == "brush":
            mask = self._random_brush_stroke_mask()
        elif damage_type == "ellipse":
            mask = self._random_ellipse_mask()
        elif damage_type == "noise":
            mask = self._random_noise_mask()
        else:  # mixed — most realistic
            masks = [
                self._random_polygon_mask(n_polygons=random.randint(1, 3)),
                self._random_brush_stroke_mask(n_strokes=random.randint(2, 5)),
                self._random_ellipse_mask(n_ellipses=random.randint(1, 3)),
            ]
            mask = np.zeros((self.size, self.size), dtype=np.uint8)
            for m in masks:
                if random.random() > 0.4:
                    mask = cv2.bitwise_or(mask, m)

        # Enforce damage ratio bounds
        ratio = mask.sum() / (255 * self.size * self.size)
        if ratio < DAMAGE_RATIO_MIN:
            # Not enough damage — add more polygons
            mask = cv2.bitwise_or(mask, self._random_polygon_mask(2))
        if ratio > DAMAGE_RATIO_MAX:
            # Too much damage — erode
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.erode(mask, kernel, iterations=3)

        return mask

    def apply_to_image(self, image: np.ndarray, mask: np.ndarray):
        """
        Apply damage mask to image.
        Masked regions are filled with grey (simulates missing stone).
        Returns: damaged_image, binary_mask (0/1 float)
        """
        damaged = image.copy()
        binary = (mask > 127).astype(np.float32)
        # Fill damaged regions with neutral grey + slight noise
        noise = np.random.randint(80, 130, image.shape, dtype=np.uint8)
        damaged[mask > 127] = noise[mask > 127]
        return damaged, binary


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────
def get_train_augmentation():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_augmentation():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# DATASET CLASSES
# ─────────────────────────────────────────────
class StyleClassificationDataset(Dataset):
    """
    Dataset for architectural style classification (Module 1).
    Directory structure expected:
        data/raw/
            Dravidian/
                img1.jpg, img2.jpg, ...
            Nagara/
                img1.jpg, ...
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_train_augmentation()
        self.class_to_idx = {c: i for i, c in enumerate(STYLE_CLASSES)}
        self.samples = []

        for style in STYLE_CLASSES:
            style_dir = self.root_dir / style
            if style_dir.exists():
                for img_path in style_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[style]))
                for img_path in style_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[style]))

        # Split 80/20 train/val deterministically
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(0.8 * len(self.samples))
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

        print(f"[StyleDataset] {split}: {len(self.samples)} samples across {len(STYLE_CLASSES)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        return augmented["image"], label


class InpaintingDataset(Dataset):
    """
    Dataset for damage detection (U-Net) and reconstruction (GAN).
    Uses original clean images + synthetically generated damage masks.
    Each __getitem__ call generates a fresh random damage mask.

    Directory structure:
        data/raw/all_monuments/
            img1.jpg, img2.jpg, ...  (clean monument images)
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_gen = DamageMaskGenerator(IMAGE_SIZE)
        self.samples = []

        img_dir = self.root_dir / "all_monuments"
        if img_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                self.samples.extend(list(img_dir.glob(ext)))

        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(0.85 * len(self.samples))
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

        self.base_transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5) if split == "train" else A.NoOp(),
            A.RandomBrightnessContrast(p=0.3) if split == "train" else A.NoOp(),
        ])
        self.normalize = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        print(f"[InpaintingDataset] {split}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply geometric augmentation
        augmented = self.base_transform(image=image)
        clean_image = augmented["image"]

        # Generate fresh synthetic damage mask
        mask_raw = self.mask_gen.generate(damage_type="mixed")
        damaged_image, binary_mask = self.mask_gen.apply_to_image(clean_image, mask_raw)

        # Normalise both images
        clean_norm = self.normalize(image=clean_image)["image"]       # [3, H, W] in [-1,1]
        damaged_norm = self.normalize(image=damaged_image)["image"]   # [3, H, W] in [-1,1]

        # Mask: 1 = damaged, 0 = intact  →  shape [1, H, W]
        mask_tensor = torch.from_numpy(
            cv2.resize(binary_mask, (IMAGE_SIZE, IMAGE_SIZE))[np.newaxis, ...]
        ).float()

        return {
            "clean": clean_norm,       # ground truth
            "damaged": damaged_norm,   # model input
            "mask": mask_tensor,       # 1=hole, 0=known
        }


# ─────────────────────────────────────────────
# DATA LOADER FACTORY
# ─────────────────────────────────────────────
def get_style_loaders(data_dir, batch_size=32, image_size=None):
    size = image_size or IMAGE_SIZE

    def _train_aug(sz):
        return A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _val_aug(sz):
        return A.Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    train_ds = StyleClassificationDataset(data_dir, split="train", transform=_train_aug(size))
    val_ds   = StyleClassificationDataset(data_dir, split="val",   transform=_val_aug(size))

    if len(train_ds) == 0:
        raise RuntimeError(
            f"\n\n[ERROR] No training images found in '{data_dir}'.\n"
            f"Expected folders like: {data_dir}/Dravidian/, {data_dir}/Nagara/, etc.\n"
            f"Each folder should contain .jpg or .png images (minimum ~50 per class).\n"
            f"Run:  python setup_data.py  to create the folder structure.\n"
            f"See README.md → 'Data Preparation' for instructions.\n"
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"\n\n[ERROR] No validation images found. Need at least 5 images per class total.\n"
            f"Add more images to '{data_dir}/<ClassName>/' folders.\n"
        )

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(batch_size, len(val_ds)),   shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader

def get_inpainting_loaders(data_dir, batch_size=8):
    train_ds = InpaintingDataset(data_dir, split="train")
    val_ds   = InpaintingDataset(data_dir, split="val")

    if len(train_ds) == 0:
        raise RuntimeError(
            f"\n\n[ERROR] No images found in '{data_dir}/all_monuments/'.\n"
            f"Place clean monument images (.jpg/.png) in that folder.\n"
            f"Run:  python setup_data.py  to create the folder structure.\n"
        )

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(batch_size, max(1, len(val_ds))), shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# QUICK TEST / VISUALISATION UTILITY
# ─────────────────────────────────────────────
def visualise_damage_samples(n=6, save_path="data/damage_samples.png"):
    """Generate a grid of damage examples for visual inspection."""
    gen = DamageMaskGenerator()
    dummy_img = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 180  # grey stone

    results = []
    for _ in range(n):
        mask = gen.generate("mixed")
        damaged, _ = gen.apply_to_image(dummy_img, mask)
        # Stack original | mask | damaged
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        row = np.hstack([dummy_img, mask_rgb, damaged])
        results.append(row)

    grid = np.vstack(results)
    Path(save_path).parent.mkdir(exist_ok=True)
    cv2.imwrite(save_path, grid)
    print(f"Saved damage sample grid → {save_path}")


if __name__ == "__main__":
    print("Running dataset module checks...")
    visualise_damage_samples(n=4, save_path="data/damage_samples.png")
    print("Dataset module OK")
