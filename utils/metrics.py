"""
utils/metrics.py — Evaluation metrics (SSIM, PSNR) and visualisation helpers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path


# ─────────────────────────────────────────────
# IMAGE QUALITY METRICS
# ─────────────────────────────────────────────
def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """SSIM between two uint8 RGB images."""
    return ssim_sk(pred, target, channel_axis=-1, data_range=255)

def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR between two uint8 RGB images."""
    return psnr_sk(target, pred, data_range=255)

def evaluate_restoration(pred_bgr, clean_bgr, mask_gray):
    """
    Compute metrics only on the damaged (restored) region.
    Returns dict with ssim, psnr (hole region and full image).
    """
    pred_rgb  = cv2.cvtColor(pred_bgr,  cv2.COLOR_BGR2RGB)
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    full_ssim = compute_ssim(pred_rgb, clean_rgb)
    full_psnr = compute_psnr(pred_rgb, clean_rgb)

    # Hole-only metrics
    hole = (mask_gray > 127)
    if hole.sum() > 0:
        pred_hole  = pred_rgb.copy();  pred_hole[~hole]  = 0
        clean_hole = clean_rgb.copy(); clean_hole[~hole] = 0
        hole_ssim = compute_ssim(pred_hole, clean_hole)
        hole_psnr = compute_psnr(pred_hole, clean_hole)
    else:
        hole_ssim = hole_psnr = 0.0

    return {
        "ssim_full": full_ssim,
        "psnr_full": full_psnr,
        "ssim_hole": hole_ssim,
        "psnr_hole": hole_psnr,
    }


# ─────────────────────────────────────────────
# CLASSIFICATION METRIC
# ─────────────────────────────────────────────
def compute_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Architectural Style Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def overlay_mask(image_bgr, mask_gray, color=(0, 0, 255), alpha=0.4):
    """Overlay damage mask on image in red (BGR)."""
    overlay = image_bgr.copy()
    overlay[mask_gray > 127] = color
    return cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

def make_comparison_grid(damaged, mask, restored, clean,
                          labels=("Damaged", "Mask", "Restored", "Original")):
    """
    Horizontally stack 4 images into a comparison grid.
    All inputs are BGR numpy uint8 arrays.
    """
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    row = np.hstack([damaged, mask_rgb, restored, clean])
    # Add column labels
    h, w = damaged.shape[:2]
    label_bar = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        cv2.putText(label_bar, label, (i * w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return np.vstack([label_bar, row])
